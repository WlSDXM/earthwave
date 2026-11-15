/*
EARTH//WAVE - audio reactive visual source for OBS
Copyright (C) 2026 Theo Kano

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <graphics/vec4.h>
#include <media-io/audio-io.h>
#include <util/bmem.h>
#include <math.h>
#include <string.h>

#define EW_PI 3.14159265358979f
#define FFT_N 4096
#define FFT_LOG 12
#define RING 16384
#define MAXB 256
#define SPAN 1024
#define WBUF 2048
#define HIST 16
#define MIX_NAME "__earthwave_output_mix__"

enum ew_style { STYLE_BARS, STYLE_SOLID, STYLE_CENTER, STYLE_SINES, STYLE_SCOPE, STYLE_ASCII };

struct ew_vert {
	float x, y;
	uint32_t c;
};

struct ew {
	obs_source_t *self;

	/* settings */
	char *audio_name;
	int style, range;
	bool mirror, gradient, gradient3, bg_on, peaks, baseline;
	uint32_t color, color2, color3, bg;
	int width, height, bars, gap, px, layers, font, sine_count, sine_thick;
	float scale, sens, smooth, decay, sine_speed, sine_size;

	/* audio in */
	obs_weak_source_t *audio_weak;
	bool mix_connected;
	float retry;
	float ring[RING];
	volatile long wpos;
	volatile long acount;
	long last_acount;
	float idle;

	/* analysis */
	bool tables;
	float win[FFT_N], cos_t[FFT_N / 2], sin_t[FFT_N / 2];
	uint16_t rev[FFT_N];
	float re[FFT_N], im[FFT_N], tbuf[FFT_N], mag[FFT_N / 2];
	float b_lo[MAXB], b_hi[MAXB], b_tilt[MAXB], raw[MAXB];
	float val[MAXB], peak[MAXB], peak_v[MAXB], hold[MAXB], hold_t[MAXB];
	int band_n, band_range;
	uint32_t band_rate;
	float agc;
	float wbuf[WBUF], wave_pk;
	int wave_st;
	float wave_phase, sine_amp;

	/* light trails: slow-falling ghost of the bars, and a short history of past lines */
	float trails;
	float trail[MAXB];
	float hist_wave[HIST][WBUF];
	int hist_st[HIST];
	float hist_pk[HIST], hist_phase[HIST], hist_amp[HIST];
	int hist_pos, hist_count;
	float hist_acc;
	const float *cw_buf; /* the waveform ew_wave_at reads from (current frame or a past one) */
	int cw_st;
	float cw_pk;
	float cycle_t;

	/* geometry built each frame */
	struct ew_vert *v;
	size_t vn, vcap;
};

/* ------------------------------------------------------------------ audio */

static void ew_push(struct ew *e, const float *l, const float *r, uint32_t frames, bool muted)
{
	/* one writer (audio thread), one reader (video thread); a torn read only costs a few stale samples */
	long w = e->wpos;
	for (uint32_t i = 0; i < frames; i++) {
		float s = 0.0f;
		if (!muted)
			s = r ? (l[i] + r[i]) * 0.5f : l[i];
		e->ring[w] = s;
		w = (w + 1) & (RING - 1);
	}
	e->wpos = w;
	e->acount = e->acount + 1;
}

static void ew_audio_cb(void *param, obs_source_t *source, const struct audio_data *ad, bool muted)
{
	UNUSED_PARAMETER(source);
	struct ew *e = param;
	if (!ad || !ad->data[0] || !ad->frames)
		return;
	ew_push(e, (const float *)ad->data[0], (const float *)ad->data[1], ad->frames, muted);
}

static void ew_mix_cb(void *param, size_t mix_idx, struct audio_data *ad)
{
	UNUSED_PARAMETER(mix_idx);
	struct ew *e = param;
	if (!ad || !ad->data[0] || !ad->frames)
		return;
	ew_push(e, (const float *)ad->data[0], (const float *)ad->data[1], ad->frames, false);
}

static void ew_detach(struct ew *e)
{
	if (e->mix_connected) {
		audio_output_disconnect(obs_get_audio(), 0, ew_mix_cb, e);
		e->mix_connected = false;
	}
	if (e->audio_weak) {
		obs_source_t *s = obs_weak_source_get_source(e->audio_weak);
		if (s) {
			obs_source_remove_audio_capture_callback(s, ew_audio_cb, e);
			obs_source_release(s);
		}
		obs_weak_source_release(e->audio_weak);
		e->audio_weak = NULL;
	}
}

static void ew_try_attach(struct ew *e)
{
	if (!e->audio_name || !*e->audio_name)
		return;

	if (strcmp(e->audio_name, MIX_NAME) == 0) {
		if (!e->mix_connected)
			e->mix_connected = audio_output_connect(obs_get_audio(), 0, NULL, ew_mix_cb, e);
		return;
	}

	if (e->audio_weak) {
		obs_source_t *alive = obs_weak_source_get_source(e->audio_weak);
		if (alive) {
			obs_source_release(alive);
			return;
		}
		obs_weak_source_release(e->audio_weak);
		e->audio_weak = NULL;
	}

	obs_source_t *s = obs_get_source_by_name(e->audio_name);
	if (!s)
		return;
	if (s != e->self) {
		obs_source_add_audio_capture_callback(s, ew_audio_cb, e);
		e->audio_weak = obs_source_get_weak_source(s);
	}
	obs_source_release(s);
}

/* --------------------------------------------------------------- analysis */

static void ew_build_tables(struct ew *e)
{
	for (int i = 0; i < FFT_N; i++) {
		e->win[i] = 0.5f - 0.5f * cosf(2.0f * EW_PI * (float)i / (float)(FFT_N - 1));
		int r = 0;
		for (int b = 0; b < FFT_LOG; b++)
			r |= ((i >> b) & 1) << (FFT_LOG - 1 - b);
		e->rev[i] = (uint16_t)r;
	}
	for (int i = 0; i < FFT_N / 2; i++) {
		e->cos_t[i] = cosf(2.0f * EW_PI * (float)i / (float)FFT_N);
		e->sin_t[i] = -sinf(2.0f * EW_PI * (float)i / (float)FFT_N);
	}
	e->tables = true;
}

/* newest n samples, oldest first */
static void ew_latest(struct ew *e, float *out, int n)
{
	long p = (e->wpos - (long)n) & (RING - 1);
	for (int i = 0; i < n; i++) {
		out[i] = e->ring[p];
		p = (p + 1) & (RING - 1);
	}
}

static void ew_fft(struct ew *e)
{
	ew_latest(e, e->tbuf, FFT_N);
	for (int i = 0; i < FFT_N; i++) {
		e->re[e->rev[i]] = e->tbuf[i] * e->win[i];
		e->im[e->rev[i]] = 0.0f;
	}
	for (int size = 2; size <= FFT_N; size <<= 1) {
		const int half = size >> 1, step = FFT_N / size;
		for (int s = 0; s < FFT_N; s += size) {
			for (int k = 0, t = 0; k < half; k++, t += step) {
				const int a = s + k, b = a + half;
				const float xr = e->re[b] * e->cos_t[t] - e->im[b] * e->sin_t[t];
				const float xi = e->re[b] * e->sin_t[t] + e->im[b] * e->cos_t[t];
				e->re[b] = e->re[a] - xr;
				e->im[b] = e->im[a] - xi;
				e->re[a] += xr;
				e->im[a] += xi;
			}
		}
	}
	for (int i = 0; i < FFT_N / 2; i++)
		e->mag[i] = sqrtf(e->re[i] * e->re[i] + e->im[i] * e->im[i]) / (float)(FFT_N / 4);
}

static void ew_build_bands(struct ew *e, uint32_t rate)
{
	static const float lo_hz[3] = {40.0f, 40.0f, 30.0f};
	static const float hi_hz[3] = {16000.0f, 8000.0f, 1500.0f};
	const int rg = e->range < 0 ? 0 : (e->range > 2 ? 2 : e->range);
	const float hz = (float)rate / (float)FFT_N;
	const float fmin = lo_hz[rg];
	float top = hi_hz[rg];
	if (top > (float)rate * 0.5f * 0.95f)
		top = (float)rate * 0.5f * 0.95f;
	const int nb = e->bars;
	for (int i = 0; i < nb; i++) {
		const float f0 = fmin * powf(top / fmin, (float)i / (float)nb);
		const float f1 = fmin * powf(top / fmin, (float)(i + 1) / (float)nb);
		e->b_lo[i] = f0 / hz;
		e->b_hi[i] = f1 / hz;
		e->b_tilt[i] = 3.5f * log2f(sqrtf(f0 * f1) / 400.0f);
	}
	e->band_n = nb;
	e->band_range = e->range;
	e->band_rate = rate;
}

static float ew_mag_at(const struct ew *e, float x)
{
	int i = (int)floorf(x);
	if (i < 0)
		i = 0;
	if (i > FFT_N / 2 - 2)
		i = FFT_N / 2 - 2;
	const float f = x - (float)i;
	return e->mag[i] * (1.0f - f) + e->mag[i + 1] * f;
}

static float ew_band_avg(const struct ew *e, float a, float b)
{
	const int nb = e->band_n > 0 ? e->band_n : 1;
	int i0 = (int)floorf(a * (float)nb);
	int i1 = (int)floorf(b * (float)nb);
	if (i1 <= i0)
		i1 = i0 + 1;
	if (i1 > nb)
		i1 = nb;
	if (i0 >= i1)
		i0 = i1 - 1;
	float s = 0.0f;
	for (int i = i0; i < i1; i++)
		s += e->val[i];
	return s / (float)(i1 - i0);
}

static void ew_analyse(struct ew *e, float dt)
{
	const int nb = e->band_n;
	float frame_max = 0.0f;

	ew_fft(e);

	for (int i = 0; i < nb; i++) {
		float m = 0.0f;
		if (e->b_hi[i] - e->b_lo[i] < 1.5f) {
			m = ew_mag_at(e, (e->b_lo[i] + e->b_hi[i]) * 0.5f);
		} else {
			int k0 = (int)ceilf(e->b_lo[i]);
			int k1 = (int)floorf(e->b_hi[i]);
			if (k1 > FFT_N / 2 - 1)
				k1 = FFT_N / 2 - 1;
			for (int k = k0; k <= k1; k++)
				if (e->mag[k] > m)
					m = e->mag[k];
		}
		const float db = 20.0f * log10f(m + 1e-9f) + e->b_tilt[i];
		float v = (db + 72.0f) / 54.0f;
		if (v < 0.0f)
			v = 0.0f;
		e->raw[i] = v;
		if (v > frame_max)
			frame_max = v;
	}

	/* slow auto gain with a floor, so silence stays silent */
	e->agc *= powf(0.5f, dt / 8.0f);
	if (frame_max > e->agc)
		e->agc = frame_max;
	if (e->agc < 0.35f)
		e->agc = 0.35f;

	const float up = 1.0f - expf(-dt / (0.008f + e->smooth * 0.25f));
	const float down = 1.0f - expf(-dt / e->decay);
	const float sc = e->scale > 1.0f ? 1.0f : e->scale;
	const float tdown = 1.0f - expf(-dt / (0.15f + e->trails * 1.5f));

	for (int i = 0; i < nb; i++) {
		float v = e->raw[i] / (e->agc * 1.04f) * e->sens;
		if (v > 1.0f)
			v = 1.0f;
		v *= sc;
		e->val[i] += (v - e->val[i]) * (v > e->val[i] ? up : down);

		/* light trail: a ghost that sinks slowly behind the bar */
		if (e->val[i] >= e->trail[i])
			e->trail[i] = e->val[i];
		else
			e->trail[i] += (e->val[i] - e->trail[i]) * tdown;

		/* falling cap for the solid style */
		if (e->val[i] >= e->peak[i]) {
			e->peak[i] = e->val[i];
			e->peak_v[i] = 0.0f;
		} else {
			e->peak_v[i] += dt * 3.5f;
			e->peak[i] -= e->peak_v[i] * dt;
			if (e->peak[i] < e->val[i])
				e->peak[i] = e->val[i];
		}

		/* floating dash for the layered style: hangs, then drifts down */
		if (e->val[i] >= e->hold[i]) {
			e->hold[i] = e->val[i];
			e->hold_t[i] = 0.0f;
		} else {
			e->hold_t[i] += dt;
			if (e->hold_t[i] > 0.35f) {
				e->hold[i] -= dt * 0.45f;
				if (e->hold[i] < e->val[i])
					e->hold[i] = e->val[i];
			}
		}
	}
}

static void ew_grab_wave(struct ew *e, float dt)
{
	ew_latest(e, e->wbuf, WBUF);
	e->wave_st = 0;
	/* lock onto a rising zero crossing so the line does not jitter */
	for (int i = 1; i < 1000; i++) {
		if (e->wbuf[i - 1] < 0.0f && e->wbuf[i] >= 0.0f && e->wbuf[i + 8] > e->wbuf[i]) {
			e->wave_st = i;
			break;
		}
	}
	float pk = 0.0f;
	for (int i = 0; i < SPAN; i++) {
		const float a = fabsf(e->wbuf[e->wave_st + i]);
		if (a > pk)
			pk = a;
	}
	e->wave_pk *= powf(0.5f, dt / 3.0f);
	if (pk > e->wave_pk)
		e->wave_pk = pk;
	if (e->wave_pk < 0.08f)
		e->wave_pk = 0.08f;
}

/* u is 0..1 across the source, returns -1..1 */
static float ew_wave_at(const struct ew *e, float u, int shift)
{
	int i = e->cw_st + shift + (int)floorf(u * (float)(SPAN - 1));
	if (i > WBUF - 3)
		i = WBUF - 3;
	if (i < 2)
		i = 2;
	const float sens = e->sens > 1.5f ? 1.5f : e->sens;
	float v = (e->cw_buf[i - 2] + e->cw_buf[i] * 2.0f + e->cw_buf[i + 2]) / 4.0f / e->cw_pk * sens;
	if (v > 1.0f)
		v = 1.0f;
	if (v < -1.0f)
		v = -1.0f;
	return v;
}

/* remember ~30 snapshots a second so lines can leave a trail */
static void ew_history(struct ew *e, float dt)
{
	if (e->trails <= 0.0f) {
		e->hist_count = 0;
		return;
	}
	e->hist_acc += dt;
	if (e->hist_acc < 1.0f / 30.0f)
		return;
	e->hist_acc = 0.0f;
	e->hist_pos = (e->hist_pos + 1) % HIST;
	if (e->style == STYLE_SCOPE || e->style == STYLE_ASCII)
		memcpy(e->hist_wave[e->hist_pos], e->wbuf, sizeof(e->wbuf));
	e->hist_st[e->hist_pos] = e->wave_st;
	e->hist_pk[e->hist_pos] = e->wave_pk;
	e->hist_phase[e->hist_pos] = e->wave_phase;
	e->hist_amp[e->hist_pos] = e->sine_amp;
	if (e->hist_count < HIST)
		e->hist_count++;
}

/* how many past snapshots to draw; "share" spreads the budget when a style draws several lines */
static int ew_ghosts(const struct ew *e, int share)
{
	if (e->trails <= 0.0f)
		return 0;
	int n = (int)roundf(e->trails * (float)(HIST - 1));
	if (share > 1)
		n = n / share + 1;
	if (n > e->hist_count)
		n = e->hist_count;
	return n;
}

static int ew_hist_index(const struct ew *e, int age)
{
	return ((e->hist_pos - (age - 1)) % HIST + HIST) % HIST;
}

static float ew_ghost_alpha(int age, int n)
{
	const float f = 1.0f - (float)age / (float)(n + 1);
	return 0.55f * f * f;
}

static void ew_tick(void *data, float seconds)
{
	struct ew *e = data;
	float dt = seconds;
	if (dt < 0.001f)
		dt = 0.001f;
	if (dt > 0.1f)
		dt = 0.1f;

	e->retry -= seconds;
	if (e->retry <= 0.0f) {
		e->retry = 1.0f;
		ew_try_attach(e);
	}

	if (!obs_source_showing(e->self))
		return;

	/* if the audio source went quiet (stopped sending), fall to zero */
	const long ac = e->acount;
	if (ac == e->last_acount) {
		e->idle += seconds;
		if (e->idle > 0.25f && e->idle < 1000.0f) {
			memset(e->ring, 0, sizeof(e->ring));
			e->idle = 1000.0f;
		}
	} else {
		e->last_acount = ac;
		e->idle = 0.0f;
	}

	if (!e->tables)
		ew_build_tables(e);

	uint32_t rate = audio_output_get_sample_rate(obs_get_audio());
	if (!rate)
		rate = 48000;
	if (e->band_n != e->bars || e->band_range != e->range || e->band_rate != rate)
		ew_build_bands(e, rate);

	ew_analyse(e, dt);

	if (e->style == STYLE_SCOPE || e->style == STYLE_ASCII)
		ew_grab_wave(e, dt);

	if (e->style == STYLE_SINES) {
		e->wave_phase += 0.9f * e->sine_speed * dt;
		if (e->wave_phase > 2000.0f * EW_PI)
			e->wave_phase -= 2000.0f * EW_PI;
		float bass = ew_band_avg(e, 0.0f, 0.2f) * 1.4f;
		if (bass > 1.0f)
			bass = 1.0f;
		e->sine_amp = 1.0f + bass * 0.9f;
	}

	ew_history(e, dt);
}

/* ---------------------------------------------------------------- drawing */

static void ew_reserve(struct ew *e, size_t extra)
{
	if (e->vn + extra <= e->vcap)
		return;
	size_t cap = e->vcap ? e->vcap * 2 : 4096;
	while (cap < e->vn + extra)
		cap *= 2;
	e->v = brealloc(e->v, cap * sizeof(struct ew_vert));
	e->vcap = cap;
}

static void ew_quad(struct ew *e, float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3,
		    uint32_t c)
{
	ew_reserve(e, 6);
	struct ew_vert *v = e->v + e->vn;
	const float xs[6] = {x0, x1, x2, x2, x1, x3};
	const float ys[6] = {y0, y1, y2, y2, y1, y3};
	for (int i = 0; i < 6; i++) {
		v[i].x = xs[i];
		v[i].y = ys[i];
		v[i].c = c;
	}
	e->vn += 6;
}

static void ew_rect(struct ew *e, float x, float y, float w, float h, uint32_t c)
{
	if (w <= 0.0f || h <= 0.0f)
		return;
	ew_quad(e, x, y, x + w, y, x, y + h, x + w, y + h, c);
}

static void ew_line(struct ew *e, float x0, float y0, float x1, float y1, float thick, uint32_t c)
{
	const float dx = x1 - x0, dy = y1 - y0;
	const float len = sqrtf(dx * dx + dy * dy);
	if (len < 0.0001f)
		return;
	const float nx = -dy / len * thick * 0.5f, ny = dx / len * thick * 0.5f;
	/* extend a touch along the line so segments overlap at the joins */
	const float ex = dx / len * thick * 0.25f, ey = dy / len * thick * 0.25f;
	ew_quad(e, x0 - ex + nx, y0 - ey + ny, x1 + ex + nx, y1 + ey + ny, x0 - ex - nx, y0 - ey - ny, x1 + ex - nx,
		y1 + ey - ny, c);
}

/* mix colour -> colour2 (-> colour3) by t (0..1), with alpha a, premultiplied, in the space OBS is rendering in */
static uint32_t ew_color(const struct ew *e, float t, float a, bool linear)
{
	uint32_t c1 = e->color, c2 = e->gradient ? e->color2 : e->color;
	if (t < 0.0f)
		t = 0.0f;
	if (t > 1.0f)
		t = 1.0f;
	if (e->gradient && e->gradient3) {
		/* three colours: first half blends 1 -> 2, second half 2 -> 3 */
		if (t < 0.5f) {
			t *= 2.0f;
		} else {
			c1 = e->color2;
			c2 = e->color3;
			t = (t - 0.5f) * 2.0f;
		}
	}
	if (a < 0.0f)
		a = 0.0f;
	if (a > 1.0f)
		a = 1.0f;
	uint32_t mixed = 0xFF000000u;
	for (int sh = 0; sh < 24; sh += 8) {
		const float p = (float)((c1 >> sh) & 0xFFu), q = (float)((c2 >> sh) & 0xFFu);
		mixed |= ((uint32_t)(p + (q - p) * t + 0.5f) & 0xFFu) << sh;
	}
	struct vec4 v;
	if (linear)
		vec4_from_rgba_srgb(&v, mixed);
	else
		vec4_from_rgba(&v, mixed);
	v.x *= a;
	v.y *= a;
	v.z *= a;
	v.w = a;
	return vec4_to_rgba(&v);
}

static uint32_t ew_plain(uint32_t rgb, bool linear)
{
	struct vec4 v;
	if (linear)
		vec4_from_rgba_srgb(&v, rgb | 0xFF000000u);
	else
		vec4_from_rgba(&v, rgb | 0xFF000000u);
	v.w = 1.0f;
	return vec4_to_rgba(&v);
}

static float ew_snap(float x, float px)
{
	return roundf(x / px) * px;
}

/* chunky pixels: solid base, scanline layers fading out toward the top, dotted dash floating above */
static void ew_draw_bars(struct ew *e, float W, float H, bool linear)
{
	const float px = (float)e->px;
	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	int rows = (int)floorf(H / px);
	if (rows < 4)
		rows = 4;
	const float gp = floorf((float)e->gap / px) * px;
	const uint32_t base = ew_color(e, 0.0f, 1.0f, linear);
	const uint32_t dash = ew_color(e, 1.0f, 1.0f, linear);

	for (int j = 0; j < n2; j++) {
		const int i = !e->mirror ? j : (j < nb ? nb - 1 - j : j - nb);
		const float x = ew_snap((float)j * W / (float)n2, px);
		const float w = ew_snap((float)(j + 1) * W / (float)n2, px) - x - gp;
		if (w <= 0.0f)
			continue;

		int n = (int)roundf(e->val[i] * (float)(rows - 2));
		if (n < 1)
			n = e->baseline ? 1 : 0;
		int solid = (int)roundf((float)n * 0.4f);
		if (solid < 1)
			solid = 1;

		if (n > 0)
			ew_rect(e, x, H - (float)solid * px, w, (float)solid * px, base);

		for (int r = solid; r < n; r++) {
			const float t = (float)r / (float)n;
			const int from_top = n - 1 - r;
			float a = 1.0f;
			if (from_top % 2 == 1) {
				a = 1.0f - (t - 0.4f) / 0.3f;
				if (a <= 0.04f)
					continue;
			}
			ew_rect(e, x, H - (float)(r + 1) * px, w, px, ew_color(e, (t - 0.4f) / 0.6f, a, linear));
		}

		if (e->trails > 0.0f) {
			const int tr = (int)roundf(e->trail[i] * (float)(rows - 2));
			for (int r = n; r < tr; r++) {
				if ((tr - 1 - r) % 2 == 1)
					continue;
				const float fade = 1.0f - (float)(r - n) / (float)(tr - n);
				ew_rect(e, x, H - (float)(r + 1) * px, w, px,
					ew_color(e, 1.0f, 0.08f + 0.45f * fade, linear));
			}
		}

		if (e->peaks) {
			const int hr = (int)roundf(e->hold[i] * (float)(rows - 2));
			if (hr > n + 1) {
				const float y = H - (float)(hr + 1) * px;
				if (w < px * 3.0f) {
					ew_rect(e, x, y, w, px, dash);
				} else {
					for (float dx = px; dx + px <= w - px; dx += px * 2.0f)
						ew_rect(e, x + dx, y, px, px, dash);
				}
			}
		}
	}
}

static void ew_draw_solid(struct ew *e, float W, float H, bool linear)
{
	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	const uint32_t c = ew_color(e, 0.0f, 1.0f, linear);
	const uint32_t cap = ew_color(e, 1.0f, 1.0f, linear);
	const uint32_t ghost = ew_color(e, 1.0f, 0.3f, linear);

	for (int j = 0; j < n2; j++) {
		const int i = !e->mirror ? j : (j < nb ? nb - 1 - j : j - nb);
		const float x = roundf((float)j * W / (float)n2);
		float w = roundf((float)(j + 1) * W / (float)n2) - x - (float)e->gap;
		if (w < 1.0f)
			w = 1.0f;
		float h = roundf(e->val[i] * (H - 4.0f));
		if (h < 2.0f)
			h = e->baseline ? 2.0f : 0.0f;
		if (e->trails > 0.0f) {
			const float th = roundf(e->trail[i] * (H - 4.0f));
			if (th > h)
				ew_rect(e, x, H - th, w, th - h, ghost);
		}
		ew_rect(e, x, H - h, w, h, c);
		if (e->peaks && e->peak[i] > 0.02f) {
			const float p = roundf(e->peak[i] * (H - 4.0f)) + 3.0f;
			ew_rect(e, x, H - p, w, 2.0f, cap);
		}
	}
}

/* bars growing up and down from the middle, bass in the centre */
static void ew_draw_center(struct ew *e, float W, float H, bool linear)
{
	const int nb = e->band_n;
	const int n2 = nb * 2;
	const float mid = floorf(H * 0.5f);
	const uint32_t c = ew_color(e, 0.0f, 1.0f, linear);
	const uint32_t ghost = ew_color(e, 0.0f, 0.3f, linear);

	for (int j = 0; j < n2; j++) {
		const int i = j < nb ? nb - 1 - j : j - nb;
		const float x = roundf((float)j * W / (float)n2);
		float w = roundf((float)(j + 1) * W / (float)n2) - x - (float)e->gap;
		if (w < 1.0f)
			w = 1.0f;
		float h = roundf(e->val[i] * (mid - 1.0f));
		if (h < 1.0f)
			h = e->baseline ? 1.0f : 0.0f;
		if (e->trails > 0.0f) {
			const float th = roundf(e->trail[i] * (mid - 1.0f));
			if (th > h) {
				ew_rect(e, x, mid - th, w, th - h, ghost);
				ew_rect(e, x, mid + h, w, th - h, ghost);
			}
		}
		ew_rect(e, x, mid - h, w, h * 2.0f, c);
	}
}

/* drifting water lines; they swell on the bass */
static void ew_sines_pass(struct ew *e, float W, float H, bool linear, float wave_phase, float sine_amp, float am)
{
	const int count = e->sine_count < 1 ? 1 : e->sine_count;
	const float k = H / 150.0f * e->sine_size;
	int steps = (int)floorf(W / 8.0f);
	if (steps < 40)
		steps = 40;

	for (int wi = 0; wi < count; wi++) {
		const float f = count > 1 ? (float)wi / (float)(count - 1) : 1.0f;
		const float y_t = 0.18f + f * 0.74f;
		const float base_y = H * (count > 1 ? 0.14f + f * 0.72f : 0.5f);
		const float depth_amp = (2.0f + y_t * 6.0f) * k;
		const float phase = wave_phase + (float)wi * 0.7f;
		const uint32_t c = ew_color(e, 1.0f - f, (0.6f + y_t * 0.7f) * 0.8f * am, linear);
		float px = 0.0f, py = 0.0f;

		for (int s = 0; s <= steps; s++) {
			const float x_t = (float)s / (float)steps;
			const float y = base_y + sinf(x_t * EW_PI * 3.0f + phase) * depth_amp * sine_amp +
					sinf(x_t * EW_PI * 7.0f + phase * 1.3f) * depth_amp * 0.3f;
			const float x = x_t * W;
			if (s > 0)
				ew_line(e, px, py, x, y, (float)e->sine_thick, c);
			px = x;
			py = y;
		}
	}
}

static void ew_draw_sines(struct ew *e, float W, float H, bool linear)
{
	const int n = ew_ghosts(e, e->sine_count > 4 ? 2 : 1);
	for (int age = n; age >= 1; age--) {
		const int h = ew_hist_index(e, age);
		ew_sines_pass(e, W, H, linear, e->hist_phase[h], e->hist_amp[h], ew_ghost_alpha(age, n));
	}
	ew_sines_pass(e, W, H, linear, e->wave_phase, e->sine_amp, 1.0f);
}

static void ew_scope_pass(struct ew *e, float W, float H, bool linear, float am)
{
	const int layers = e->layers < 1 ? 1 : e->layers;

	for (int l = layers - 1; l >= 0; l--) {
		const float depth = layers > 1 ? (float)l / (float)(layers - 1) : 0.0f;
		const float base = layers > 1 ? H * 0.64f - depth * H * 0.28f : H * 0.5f;
		const float amp = (layers > 1 ? H * 0.33f : H * 0.5f - 3.0f) * (1.0f - depth * 0.35f);
		const uint32_t c = ew_color(e, depth, (0.9f - depth * 0.6f) * am, linear);
		const int shift = (int)floorf(depth * 12.0f);
		float px = 0.0f, py = 0.0f;

		for (float x = 0.0f; x <= W; x += 2.0f) {
			const float y = base - ew_wave_at(e, x / W, shift) * amp;
			if (x > 0.0f)
				ew_line(e, px, py, x, y, 2.0f, c);
			px = x;
			py = y;
		}
	}
}

/* point ew_wave_at at a past snapshot (age >= 1) or the current frame (age 0) */
static void ew_use_wave(struct ew *e, int age)
{
	if (age > 0) {
		const int h = ew_hist_index(e, age);
		e->cw_buf = e->hist_wave[h];
		e->cw_st = e->hist_st[h];
		e->cw_pk = e->hist_pk[h];
	} else {
		e->cw_buf = e->wbuf;
		e->cw_st = e->wave_st;
		e->cw_pk = e->wave_pk;
	}
}

static void ew_draw_scope(struct ew *e, float W, float H, bool linear)
{
	const int n = ew_ghosts(e, e->layers);
	for (int age = n; age >= 1; age--) {
		ew_use_wave(e, age);
		ew_scope_pass(e, W, H, linear, ew_ghost_alpha(age, n));
	}
	ew_use_wave(e, 0);
	ew_scope_pass(e, W, H, linear, 1.0f);
}

/* 5x7 glyphs for - _ / \ | (one byte per row, bit 4 = left column) */
static const uint8_t ew_glyphs[5][7] = {
	{0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F},
	{0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}, {0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01},
	{0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
};

static void ew_glyph(struct ew *e, int g, float x, float y, float u, uint32_t c)
{
	for (int r = 0; r < 7; r++) {
		const uint8_t bits = ew_glyphs[g][r];
		for (int col = 0; col < 5; col++)
			if (bits & (0x10 >> col))
				ew_rect(e, x + (float)col * u, y + (float)r * u, u, u, c);
	}
}

static float ew_ascii_pos(const struct ew *e, int c, int cols, int rows)
{
	float p = (0.5f - ew_wave_at(e, (float)c / (float)cols, 0) * 0.5f) * (float)rows;
	if (p < 0.0f)
		p = 0.0f;
	if (p > (float)rows - 0.001f)
		p = (float)rows - 0.001f;
	return p;
}

static void ew_ascii_pass(struct ew *e, float W, float H, bool linear, float am)
{
	float u = roundf((float)e->font / 8.0f);
	if (u < 1.0f)
		u = 1.0f;
	const float cw = u * 6.0f, lh = u * 8.0f;
	const int cols = (int)floorf(W / cw);
	int rows = (int)floorf(H / lh);
	if (rows < 3)
		rows = 3;
	if (cols < 2)
		return;
	const float top = H - (float)rows * lh;
	const uint32_t c = ew_color(e, 0.0f, am, linear);
	float y0 = ew_ascii_pos(e, 0, cols, rows);

	for (int col = 0; col < cols; col++) {
		const float y1 = ew_ascii_pos(e, col + 1, cols, rows);
		const int r0 = (int)floorf(y0), r1 = (int)floorf(y1);
		const float x = (float)col * cw;
		if (r1 == r0) {
			ew_glyph(e, (y0 - (float)r0) < 0.6f ? 0 : 1, x, top + (float)r0 * lh, u, c);
		} else {
			const int step = r1 < r0 ? -1 : 1;
			ew_glyph(e, r1 < r0 ? 2 : 3, x, top + (float)r0 * lh, u, c);
			for (int r = r0 + step; r != r1; r += step)
				ew_glyph(e, 4, x, top + (float)r * lh, u, c);
		}
		y0 = y1;
	}
}

static void ew_draw_ascii(struct ew *e, float W, float H, bool linear)
{
	const int n = ew_ghosts(e, 3);
	for (int age = n; age >= 1; age--) {
		ew_use_wave(e, age);
		ew_ascii_pass(e, W, H, linear, ew_ghost_alpha(age, n));
	}
	ew_use_wave(e, 0);
	ew_ascii_pass(e, W, H, linear, 1.0f);
}

static void ew_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct ew *e = data;
	const float W = (float)e->width, H = (float)e->height;
	const bool linear = gs_get_linear_srgb();

	if (e->band_n <= 0 || W < 1.0f || H < 1.0f)
		return;

	e->vn = 0;
	if (e->bg_on)
		ew_rect(e, 0.0f, 0.0f, W, H, ew_plain(e->bg, linear));

	switch (e->style) {
	case STYLE_SOLID:
		ew_draw_solid(e, W, H, linear);
		break;
	case STYLE_CENTER:
		ew_draw_center(e, W, H, linear);
		break;
	case STYLE_SINES:
		ew_draw_sines(e, W, H, linear);
		break;
	case STYLE_SCOPE:
		ew_draw_scope(e, W, H, linear);
		break;
	case STYLE_ASCII:
		ew_draw_ascii(e, W, H, linear);
		break;
	default:
		ew_draw_bars(e, W, H, linear);
		break;
	}

	if (!e->vn)
		return;

	gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
	gs_eparam_t *color = gs_effect_get_param_by_name(solid, "color");
	gs_technique_t *tech = gs_effect_get_technique(solid, "SolidColored");
	struct vec4 white;
	vec4_set(&white, 1.0f, 1.0f, 1.0f, 1.0f);
	gs_effect_set_vec4(color, &white);

	const bool previous = gs_framebuffer_srgb_enabled();
	gs_enable_framebuffer_srgb(linear);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);

	gs_render_start(true);
	for (size_t i = 0; i < e->vn; i++) {
		gs_vertex2f(e->v[i].x, e->v[i].y);
		gs_color(e->v[i].c);
	}
	gs_render_stop(GS_TRIS);

	gs_technique_end_pass(tech);
	gs_technique_end(tech);

	gs_blend_state_pop();
	gs_enable_framebuffer_srgb(previous);
}

/* ------------------------------------------------------------- properties */

static bool ew_add_audio_source(void *param, obs_source_t *source)
{
	obs_property_t *list = param;
	if (obs_source_get_output_flags(source) & OBS_SOURCE_AUDIO) {
		const char *name = obs_source_get_name(source);
		if (name && *name)
			obs_property_list_add_string(list, name, name);
	}
	return true;
}

static void ew_show(obs_properties_t *props, const char *name, bool visible)
{
	obs_property_t *p = obs_properties_get(props, name);
	if (p)
		obs_property_set_visible(p, visible);
}

static bool ew_style_changed(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	UNUSED_PARAMETER(property);
	const int style = (int)obs_data_get_int(settings, "style");
	const bool bars = style == STYLE_BARS || style == STYLE_SOLID || style == STYLE_CENTER;
	const bool grad = obs_data_get_bool(settings, "gradient");

	ew_show(props, "mirror", style == STYLE_BARS || style == STYLE_SOLID);
	ew_show(props, "range", bars);
	ew_show(props, "color2", grad && style != STYLE_CENTER && style != STYLE_ASCII);
	ew_show(props, "gradient", style != STYLE_CENTER && style != STYLE_ASCII);
	ew_show(props, "gradient3", grad && style != STYLE_CENTER && style != STYLE_ASCII);
	ew_show(props, "color3",
		grad && obs_data_get_bool(settings, "gradient3") && style != STYLE_CENTER && style != STYLE_ASCII);
	ew_show(props, "bg", obs_data_get_bool(settings, "bg_on"));
	ew_show(props, "scale", bars);
	ew_show(props, "smooth", bars || style == STYLE_SINES);
	ew_show(props, "decay", bars || style == STYLE_SINES);
	ew_show(props, "bars", bars);
	ew_show(props, "gap", bars);
	ew_show(props, "px", style == STYLE_BARS);
	ew_show(props, "peaks", style == STYLE_BARS || style == STYLE_SOLID);
	ew_show(props, "baseline", bars);
	ew_show(props, "layers", style == STYLE_SCOPE);
	ew_show(props, "font", style == STYLE_ASCII);
	ew_show(props, "sine_count", style == STYLE_SINES);
	ew_show(props, "sine_speed", style == STYLE_SINES);
	ew_show(props, "sine_thick", style == STYLE_SINES);
	ew_show(props, "sine_size", style == STYLE_SINES);
	return true;
}

static obs_properties_t *ew_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();
	obs_property_t *p;

	p = obs_properties_add_list(props, "audio_source", "Audio Source", OBS_COMBO_TYPE_LIST,
				    OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p, "None", "");
	obs_property_list_add_string(p, "Stream mix (Track 1)", MIX_NAME);
	obs_enum_sources(ew_add_audio_source, p);

	p = obs_properties_add_list(props, "style", "Style", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, "Bars (layered)", STYLE_BARS);
	obs_property_list_add_int(p, "Bars (solid)", STYLE_SOLID);
	obs_property_list_add_int(p, "Bars (centred)", STYLE_CENTER);
	obs_property_list_add_int(p, "Sine waves", STYLE_SINES);
	obs_property_list_add_int(p, "Oscilloscope", STYLE_SCOPE);
	obs_property_list_add_int(p, "Oscilloscope ASCII", STYLE_ASCII);
	obs_property_set_modified_callback(p, ew_style_changed);

	obs_properties_add_bool(props, "mirror", "Mirror");

	p = obs_properties_add_list(props, "range", "Frequency Detail", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(p, "Wide Spectrum", 0);
	obs_property_list_add_int(p, "Mid (no top end)", 1);
	obs_property_list_add_int(p, "Bass Focus", 2);

	obs_properties_add_color(props, "color", "Color");
	p = obs_properties_add_bool(props, "gradient", "Use Gradient");
	obs_property_set_modified_callback(p, ew_style_changed);
	obs_properties_add_color(props, "color2", "Gradient Color 2");
	p = obs_properties_add_bool(props, "gradient3", "Use Third Color");
	obs_property_set_modified_callback(p, ew_style_changed);
	obs_properties_add_color(props, "color3", "Gradient Color 3");
	p = obs_properties_add_bool(props, "bg_on", "Use Background Color");
	obs_property_set_modified_callback(p, ew_style_changed);
	obs_properties_add_color(props, "bg", "Background Color");

	obs_properties_add_int(props, "width", "Width", 16, 7680, 1);
	obs_properties_add_int(props, "height", "Height", 16, 4320, 1);

	obs_properties_add_float_slider(props, "scale", "Height Scale", 0.1, 1.0, 0.01);
	obs_properties_add_float_slider(props, "sens", "Sensitivity", 0.1, 3.0, 0.01);
	obs_properties_add_float_slider(props, "smooth", "Smoothing", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(props, "decay", "Decay", 0.02, 0.8, 0.01);
	obs_properties_add_float_slider(props, "trails", "Light Trails", 0.0, 1.0, 0.01);

	obs_properties_add_int_slider(props, "bars", "Bars", 4, 128, 1);
	obs_properties_add_int_slider(props, "gap", "Gap", 0, 12, 1);
	obs_properties_add_int_slider(props, "px", "Pixel Size", 1, 10, 1);
	obs_properties_add_bool(props, "peaks", "Peak Dashes");
	obs_properties_add_bool(props, "baseline", "Show Baseline When Silent");

	obs_properties_add_int_slider(props, "layers", "Scope Layers", 1, 12, 1);
	obs_properties_add_int_slider(props, "font", "ASCII Size", 8, 48, 1);

	obs_properties_add_int_slider(props, "sine_count", "Waves", 1, 12, 1);
	obs_properties_add_float_slider(props, "sine_speed", "Wave Speed", 0.0, 4.0, 0.05);
	obs_properties_add_int_slider(props, "sine_thick", "Wave Thickness", 1, 8, 1);
	obs_properties_add_float_slider(props, "sine_size", "Wave Size", 0.2, 3.0, 0.05);

	return props;
}

static void ew_defaults(obs_data_t *s)
{
	struct obs_video_info ovi;
	int width = 1920;
	if (obs_get_video_info(&ovi) && ovi.base_width)
		width = (int)ovi.base_width;

	obs_data_set_default_string(s, "audio_source", "");
	obs_data_set_default_int(s, "style", STYLE_BARS);
	obs_data_set_default_bool(s, "mirror", true);
	obs_data_set_default_int(s, "range", 0);
	obs_data_set_default_int(s, "color", 0xFFFFFFFF);
	obs_data_set_default_bool(s, "gradient", true);
	obs_data_set_default_int(s, "color2", 0xFFD2A92A); /* #2aa9d2, stored as ABGR */
	obs_data_set_default_bool(s, "gradient3", false);
	obs_data_set_default_int(s, "color3", 0xFFFF3C7A); /* #7a3cff */
	obs_data_set_default_bool(s, "bg_on", false);
	obs_data_set_default_int(s, "bg", 0xFF000000);
	obs_data_set_default_int(s, "width", width);
	obs_data_set_default_int(s, "height", 200);
	obs_data_set_default_double(s, "scale", 1.0);
	obs_data_set_default_double(s, "sens", 1.0);
	obs_data_set_default_double(s, "smooth", 0.1);
	obs_data_set_default_double(s, "decay", 0.14);
	obs_data_set_default_double(s, "trails", 0.0);
	obs_data_set_default_int(s, "bars", 40);
	obs_data_set_default_int(s, "gap", 2);
	obs_data_set_default_int(s, "px", 3);
	obs_data_set_default_bool(s, "peaks", true);
	obs_data_set_default_bool(s, "baseline", false);
	obs_data_set_default_int(s, "layers", 1);
	obs_data_set_default_int(s, "font", 16);
	obs_data_set_default_int(s, "sine_count", 5);
	obs_data_set_default_double(s, "sine_speed", 1.0);
	obs_data_set_default_int(s, "sine_thick", 2);
	obs_data_set_default_double(s, "sine_size", 1.0);
}

static int ew_clampi(long long v, int lo, int hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return (int)v;
}

static float ew_clampf(double v, float lo, float hi)
{
	const float f = (float)v;
	if (f < lo)
		return lo;
	if (f > hi)
		return hi;
	return f;
}

static void ew_update(void *data, obs_data_t *s)
{
	struct ew *e = data;
	const char *name = obs_data_get_string(s, "audio_source");

	if (!e->audio_name || strcmp(e->audio_name, name ? name : "") != 0) {
		ew_detach(e);
		bfree(e->audio_name);
		e->audio_name = bstrdup(name ? name : "");
		memset(e->ring, 0, sizeof(e->ring));
		e->retry = 0.0f;
	}

	const int old_style = e->style;
	e->style = ew_clampi(obs_data_get_int(s, "style"), STYLE_BARS, STYLE_ASCII);
	if (e->style != old_style)
		e->hist_count = 0; /* past snapshots belong to the old style */
	e->trails = ew_clampf(obs_data_get_double(s, "trails"), 0.0f, 1.0f);
	e->mirror = obs_data_get_bool(s, "mirror");
	e->range = ew_clampi(obs_data_get_int(s, "range"), 0, 2);
	e->color = (uint32_t)obs_data_get_int(s, "color");
	e->gradient = obs_data_get_bool(s, "gradient");
	e->color2 = (uint32_t)obs_data_get_int(s, "color2");
	e->gradient3 = obs_data_get_bool(s, "gradient3");
	e->color3 = (uint32_t)obs_data_get_int(s, "color3");
	e->bg_on = obs_data_get_bool(s, "bg_on");
	e->bg = (uint32_t)obs_data_get_int(s, "bg");
	e->width = ew_clampi(obs_data_get_int(s, "width"), 16, 7680);
	e->height = ew_clampi(obs_data_get_int(s, "height"), 16, 4320);
	e->scale = ew_clampf(obs_data_get_double(s, "scale"), 0.05f, 1.0f);
	e->sens = ew_clampf(obs_data_get_double(s, "sens"), 0.05f, 3.0f);
	e->smooth = ew_clampf(obs_data_get_double(s, "smooth"), 0.0f, 1.0f);
	e->decay = ew_clampf(obs_data_get_double(s, "decay"), 0.01f, 2.0f);
	e->bars = ew_clampi(obs_data_get_int(s, "bars"), 4, MAXB / 2);
	e->gap = ew_clampi(obs_data_get_int(s, "gap"), 0, 64);
	e->px = ew_clampi(obs_data_get_int(s, "px"), 1, 32);
	e->peaks = obs_data_get_bool(s, "peaks");
	e->baseline = obs_data_get_bool(s, "baseline");
	e->layers = ew_clampi(obs_data_get_int(s, "layers"), 1, 12);
	e->font = ew_clampi(obs_data_get_int(s, "font"), 8, 96);
	e->sine_count = ew_clampi(obs_data_get_int(s, "sine_count"), 1, 12);
	e->sine_speed = ew_clampf(obs_data_get_double(s, "sine_speed"), 0.0f, 4.0f);
	e->sine_thick = ew_clampi(obs_data_get_int(s, "sine_thick"), 1, 8);
	e->sine_size = ew_clampf(obs_data_get_double(s, "sine_size"), 0.2f, 3.0f);
}

/* ---------------------------------------------------------------- plumbing */

static const char *ew_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return "EARTH//WAVE";
}

static void *ew_create(obs_data_t *settings, obs_source_t *source)
{
	struct ew *e = bzalloc(sizeof(struct ew));
	e->self = source;
	e->agc = 0.35f;
	e->wave_pk = 0.1f;
	e->sine_amp = 1.0f;
	e->idle = 1000.0f;
	e->cw_buf = e->wbuf;
	e->cw_pk = 0.1f;
	ew_update(e, settings);
	return e;
}

static void ew_destroy(void *data)
{
	struct ew *e = data;
	ew_detach(e);
	bfree(e->audio_name);
	bfree(e->v);
	bfree(e);
}

static uint32_t ew_get_width(void *data)
{
	const struct ew *e = data;
	return (uint32_t)e->width;
}

static uint32_t ew_get_height(void *data)
{
	const struct ew *e = data;
	return (uint32_t)e->height;
}

struct obs_source_info earthwave_source_info = {
	.id = "earthwave_source",
	.type = OBS_SOURCE_TYPE_INPUT,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
	.get_name = ew_get_name,
	.create = ew_create,
	.destroy = ew_destroy,
	.update = ew_update,
	.get_defaults = ew_defaults,
	.get_properties = ew_properties,
	.get_width = ew_get_width,
	.get_height = ew_get_height,
	.video_tick = ew_tick,
	.video_render = ew_render,
	.icon_type = OBS_ICON_TYPE_AUDIO_OUTPUT,
};
