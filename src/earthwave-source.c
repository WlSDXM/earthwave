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
#define RIDGE 16
#define MIX_NAME "__earthwave_output_mix__"

enum ew_style {
	STYLE_BARS,
	STYLE_SOLID,
	STYLE_CENTER,
	STYLE_SINES,
	STYLE_SCOPE,
	STYLE_ASCII,
	STYLE_LED,
	STYLE_RIDGE,
	STYLE_PIXWAVE,
	STYLE_CURVE,
	STYLE_WATER, /* retired */
	STYLE_DITHER,
	STYLE_SEA, /* retired: kept so saved style numbers still match */
	STYLE_OCEAN,
	STYLE_GLOW, /* retired */
	STYLE_REFLECT,
	STYLE_SKYLINE,
	STYLE_KANA,
	STYLE_RAIN,
	STYLE_LAST = STYLE_RAIN
};

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
	float scale, sens, smooth, decay, sine_speed, sine_size, sine_react, length;

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
	float ridge[RIDGE][MAXB]; /* past spectra for the ridgeline style, newest at ridge_pos */
	int ridge_pos;
	float ridge_acc;
	float energy, water_t;           /* slow overall loudness, and the sea's own clock */
	float rain_t;                    /* the rain's own clock: runs faster when the music is louder */
	double clock;                    /* seconds since the source started, for the sea styles */
	float beat, avg_bass, prev_bass; /* kick detector: beat jumps to 1 on a kick and decays */
	float ox;                        /* x offset applied to everything drawn (for the Length slider) */
	const float *cw_buf;             /* the waveform ew_wave_at reads from (current frame or a past one) */
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

/* slow overall loudness, used by the water style for the sea level */
static void ew_energy(struct ew *e, float dt)
{
	float avg = 0.0f;
	for (int i = 0; i < e->band_n; i++)
		avg += e->val[i];
	avg /= (float)(e->band_n > 0 ? e->band_n : 1);
	const float k = 1.0f - expf(-dt / (avg > e->energy ? 0.15f : 0.7f));
	e->energy += (avg - e->energy) * k;
	e->water_t += dt * (0.6f + e->energy * 2.2f);
	if (e->water_t > 4000.0f * EW_PI)
		e->water_t -= 4000.0f * EW_PI;
	e->clock += (double)dt;
	e->rain_t += dt * (0.6f + e->energy * 2.5f);
	if (e->rain_t > 100000.0f)
		e->rain_t -= 100000.0f;

	/* kick = bass well above its recent average, or a sudden jump in bass */
	const float bass = ew_band_avg(e, 0.0f, 0.2f);
	e->avg_bass += (bass - e->avg_bass) * (1.0f - expf(-dt / 0.33f));
	const float rise = bass - e->prev_bass;
	e->prev_bass = bass;
	const bool hit = (bass > e->avg_bass * 1.25f && bass > 0.25f) ||
			 (rise > 0.045f && bass > 0.3f && e->beat < 0.5f);
	e->beat = hit ? 1.0f : e->beat * powf(0.88f, dt * 60.0f);
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
	if (e->style == STYLE_SCOPE || e->style == STYLE_ASCII || e->style == STYLE_PIXWAVE)
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
	ew_energy(e, dt);

	if (e->style == STYLE_SCOPE || e->style == STYLE_ASCII || e->style == STYLE_PIXWAVE)
		ew_grab_wave(e, dt);

	if (e->style == STYLE_RIDGE) {
		e->ridge_acc += dt;
		if (e->ridge_acc >= 1.0f / 24.0f) {
			e->ridge_acc = 0.0f;
			e->ridge_pos = (e->ridge_pos + 1) % RIDGE;
			memcpy(e->ridge[e->ridge_pos], e->val, sizeof(e->val));
		}
	}

	if (e->style == STYLE_SINES) {
		float bass = ew_band_avg(e, 0.0f, 0.2f) * 1.4f;
		if (bass > 1.0f)
			bass = 1.0f;
		/* the drift surges forward on the bass, so the lines visibly push with the beat */
		e->wave_phase += 0.9f * e->sine_speed * dt * (1.0f + bass * 2.5f * e->sine_react);
		if (e->wave_phase > 2000.0f * EW_PI)
			e->wave_phase -= 2000.0f * EW_PI;
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
		v[i].x = xs[i] + e->ox;
		v[i].y = ys[i];
		v[i].c = c;
	}
	e->vn += 6;
}

/* quad with a colour per corner (top-left, top-right, bottom-left, bottom-right) */
static void ew_quad_grad(struct ew *e, float x0, float y0, float x1, float y1, float x2, float y2, float x3, float y3,
			 uint32_t c_top, uint32_t c_bottom)
{
	ew_reserve(e, 6);
	struct ew_vert *v = e->v + e->vn;
	const float xs[6] = {x0, x1, x2, x2, x1, x3};
	const float ys[6] = {y0, y1, y2, y2, y1, y3};
	const uint32_t cs[6] = {c_top, c_top, c_bottom, c_bottom, c_top, c_bottom};
	for (int i = 0; i < 6; i++) {
		v[i].x = xs[i] + e->ox;
		v[i].y = ys[i];
		v[i].c = cs[i];
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
static void ew_sines_pass(struct ew *e, float W, float H, bool linear, float wave_phase, float am)
{
	const int count = e->sine_count < 1 ? 1 : e->sine_count;
	const float k = H / 150.0f * e->sine_size;
	int steps = (int)floorf(W / 6.0f);
	if (steps < 40)
		steps = 40;

	for (int wi = 0; wi < count; wi++) {
		const float f = count > 1 ? (float)wi / (float)(count - 1) : 1.0f;
		const float y_t = 0.18f + f * 0.74f;
		const float base_y = H * (count > 1 ? 0.2f + f * 0.6f : 0.5f);
		const float depth_amp = (2.0f + y_t * 6.0f) * k;
		const float phase = wave_phase + (float)wi * 0.7f;
		const uint32_t c = ew_color(e, 1.0f - f, (0.6f + y_t * 0.7f) * 0.8f * am, linear);

		/* each line listens to its own slice of the spectrum: front line = bass, back line = highs */
		const float slice = 1.0f / (float)count;
		const float lo = (1.0f - f) * (1.0f - slice);
		float energy = ew_band_avg(e, lo, lo + slice);
		if (energy > 1.0f)
			energy = 1.0f;
		float amp = depth_amp * (0.35f + energy * 3.0f * e->sine_react) * e->sine_amp;
		float ripple = depth_amp * 0.9f * energy * e->sine_react;
		/* keep the whole swing inside the source so peaks are never cut flat */
		float room = (base_y < H - base_y ? base_y : H - base_y) - 2.0f;
		if (room < 1.0f)
			room = 1.0f;
		const float swing = amp * 1.3f + ripple;
		if (swing > room) {
			amp *= room / swing;
			ripple *= room / swing;
		}
		float px = 0.0f, py = 0.0f;

		for (int s = 0; s <= steps; s++) {
			const float x_t = (float)s / (float)steps;
			float y = base_y + sinf(x_t * EW_PI * 3.0f + phase) * amp +
				  sinf(x_t * EW_PI * 7.0f + phase * 1.3f) * amp * 0.3f +
				  sinf(x_t * EW_PI * 23.0f - phase * 3.1f) * ripple;
			if (y < 1.0f)
				y = 1.0f;
			if (y > H - 1.0f)
				y = H - 1.0f;
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
		ew_sines_pass(e, W, H, linear, e->hist_phase[h], ew_ghost_alpha(age, n));
	}
	ew_sines_pass(e, W, H, linear, e->wave_phase, 1.0f);
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
enum ew_glyph_id {
	G_DASH,
	G_UNDER,
	G_SLASH,
	G_BSLASH,
	G_PIPE,
	G_TILDE,
	G_APPROX,
	G_DOT,
	G_EQUAL,
	G_PLUS,
	G_STAR,
	G_O,
	G_SHADE1,
	G_SHADE2,
	G_COUNT
};

/* 5x7 glyphs (one byte per row, bit 4 = left column); the two shade glyphs are drawn as patterns instead */
static const uint8_t ew_glyphs[G_COUNT][7] = {
	{0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F},
	{0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}, {0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01},
	{0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, {0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00},
	{0x00, 0x08, 0x15, 0x02, 0x08, 0x15, 0x02}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00},
	{0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00}, {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00},
	{0x00, 0x15, 0x0E, 0x1F, 0x0E, 0x15, 0x00}, {0x00, 0x00, 0x0E, 0x11, 0x11, 0x0E, 0x00},
	{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
};

static void ew_glyph(struct ew *e, int g, float x, float y, float u, uint32_t c)
{
	if (g == G_SHADE1 || g == G_SHADE2) {
		/* light and medium shade blocks: a dot pattern over the whole 6x8 cell */
		for (int r = 0; r < 8; r++) {
			for (int col = 0; col < 6; col++) {
				const bool on = g == G_SHADE1 ? (r % 2 == 0 && (col + r) % 4 == 0)
							      : ((col + r) % 2 == 0);
				if (on)
					ew_rect(e, x + (float)col * u, y + (float)r * u, u, u, c);
			}
		}
		return;
	}
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
			ew_glyph(e, (y0 - (float)r0) < 0.6f ? G_DASH : G_UNDER, x, top + (float)r0 * lh, u, c);
		} else {
			const int step = r1 < r0 ? -1 : 1;
			ew_glyph(e, r1 < r0 ? G_SLASH : G_BSLASH, x, top + (float)r0 * lh, u, c);
			for (int r = r0 + step; r != r1; r += step)
				ew_glyph(e, G_PIPE, x, top + (float)r * lh, u, c);
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

/* LED meter: stacked segments, coloured by how high they sit (base colour at the bottom, last colour at the top) */
static void ew_draw_led(struct ew *e, float W, float H, bool linear)
{
	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	const float seg = (float)e->px * 2.0f;
	const float pitch = seg + (float)(e->px > 2 ? e->px : 2);
	int rows = (int)floorf(H / pitch);
	if (rows < 3)
		rows = 3;

	for (int j = 0; j < n2; j++) {
		const int i = !e->mirror ? j : (j < nb ? nb - 1 - j : j - nb);
		const float x = roundf((float)j * W / (float)n2);
		float w = roundf((float)(j + 1) * W / (float)n2) - x - (float)(e->gap > 1 ? e->gap : 1);
		if (w < 1.0f)
			w = 1.0f;
		int n = (int)roundf(e->val[i] * (float)rows);
		if (n < 1)
			n = e->baseline ? 1 : 0;

		for (int r = 0; r < n; r++)
			ew_rect(e, x, H - (float)(r + 1) * pitch + (pitch - seg), w, seg,
				ew_color(e, (float)r / (float)(rows - 1), 1.0f, linear));

		if (e->trails > 0.0f) {
			const int tr = (int)roundf(e->trail[i] * (float)rows);
			for (int r = n; r < tr; r++)
				ew_rect(e, x, H - (float)(r + 1) * pitch + (pitch - seg), w, seg,
					ew_color(e, (float)r / (float)(rows - 1), 0.28f, linear));
		}

		if (e->peaks) {
			const int hr = (int)roundf(e->hold[i] * (float)rows);
			if (hr > n && hr < rows + 1)
				ew_rect(e, x, H - (float)(hr + 1) * pitch + (pitch - seg), w, seg,
					ew_color(e, (float)hr / (float)(rows - 1), 1.0f, linear));
		}
	}
}

/* value of the (optionally mirrored) spectrum at point j of n2, from a given set of bands */
static float ew_spec(const struct ew *e, const float *vals, int j, int n2)
{
	const int nb = e->band_n;
	if (j < 0)
		j = 0;
	if (j > n2 - 1)
		j = n2 - 1;
	const int i = !e->mirror ? j : (j < nb ? nb - 1 - j : j - nb);
	return vals[i];
}

/* smooth (Catmull-Rom) reading of the spectrum at fractional point p */
static float ew_spec_smooth(const struct ew *e, const float *vals, float p, int n2)
{
	const int j = (int)floorf(p);
	const float t = p - (float)j;
	const float a = ew_spec(e, vals, j - 1, n2), b = ew_spec(e, vals, j, n2);
	const float c = ew_spec(e, vals, j + 1, n2), d = ew_spec(e, vals, j + 2, n2);
	float v = 0.5f * ((2.0f * b) + (-a + c) * t + (2.0f * a - 5.0f * b + 4.0f * c - d) * t * t +
			  (-a + 3.0f * b - 3.0f * c + d) * t * t * t);
	if (v < 0.0f)
		v = 0.0f;
	if (v > 1.0f)
		v = 1.0f;
	return v;
}

/* ridgeline: the spectrum as a line, with older moments stacked up behind it and fading away */
static void ew_draw_ridge(struct ew *e, float W, float H, bool linear)
{
	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	int lines = 4 + e->layers;
	if (lines > RIDGE)
		lines = RIDGE;
	const int sub = 4;
	const int pts = (n2 - 1) * sub;

	for (int k = lines - 1; k >= 0; k--) {
		const float *vals = e->ridge[((e->ridge_pos - k) % RIDGE + RIDGE) % RIDGE];
		const float depth = (float)k / (float)(lines - 1);
		const float base = H - 2.0f - depth * H * 0.55f;
		const float amp = H * 0.42f * (1.0f - depth * 0.45f);
		const float span = W * (1.0f - depth * 0.3f);
		const float x0 = (W - span) * 0.5f;
		const uint32_t c = ew_color(e, depth, 1.0f - depth * 0.8f, linear);
		float px = 0.0f, py = 0.0f;
		bool pen = false;

		for (int s = 0; s <= pts; s++) {
			const float v = ew_spec_smooth(e, vals, (float)s / (float)sub, n2);
			const float x = x0 + span * (float)s / (float)pts;
			const float y = base - v * amp;
			/* nothing is drawn where the spectrum is silent, so there is no resting line */
			const bool on = v > 0.012f;
			if (pen && on)
				ew_line(e, px, py, x, y, k == 0 ? 2.0f : 1.5f, c);
			px = x;
			py = y;
			pen = on;
		}
	}
}

/* spectrum as a smooth glowing outline with a soft fill underneath */
static void ew_curve_pass(struct ew *e, const float *vals, float W, float H, bool linear, float am, bool fill)
{
	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	const int sub = 4;
	const int pts = (n2 - 1) * sub;
	const float amp = H - 4.0f;
	const uint32_t line = ew_color(e, 0.0f, am, linear);
	const uint32_t clear = ew_color(e, 0.0f, 0.0f, linear);
	float px = 0.0f, pv = 0.0f;

	for (int s = 0; s <= pts; s++) {
		const float v = ew_spec_smooth(e, vals, (float)s / (float)sub, n2);
		const float x = W * (float)s / (float)pts;
		if (s > 0 && (v > 0.012f || pv > 0.012f)) {
			const float y0 = H - pv * amp, y1 = H - v * amp;
			if (fill) {
				const float mid = (pv + v) * 0.5f;
				ew_quad_grad(e, px, y0, x, y1, px, H, x, H,
					     ew_color(e, 0.4f + mid * 0.6f, 0.55f * am, linear), clear);
			}
			ew_line(e, px, y0, x, y1, 2.0f, line);
		}
		px = x;
		pv = v;
	}
}

static void ew_draw_curve(struct ew *e, float W, float H, bool linear)
{
	if (e->trails > 0.0f)
		ew_curve_pass(e, e->trail, W, H, linear, 0.35f, false);
	ew_curve_pass(e, e->val, W, H, linear, 1.0f, true);
}

/* the waveform as chunky pixel columns mirrored around the middle, tips broken into layers */
static void ew_pixwave_pass(struct ew *e, float W, float H, bool linear, float am)
{
	const float px = (float)e->px;
	const float colw = px * 2.0f;
	const float gp = px;
	const int cols = (int)floorf(W / (colw + gp));
	const float mid = ew_snap(H * 0.5f, px);
	int rows = (int)floorf(H * 0.5f / px) - 1;
	if (rows < 2)
		rows = 2;
	if (cols < 2)
		return;
	const float x0 = floorf((W - (float)cols * (colw + gp) + gp) * 0.5f);

	for (int c = 0; c < cols; c++) {
		float pk = 0.0f;
		for (int k = 0; k < 4; k++) {
			const float a = fabsf(ew_wave_at(e, ((float)c + (float)k * 0.25f) / (float)cols, 0));
			if (a > pk)
				pk = a;
		}
		const int n = (int)roundf(pk * (float)rows);
		const float x = x0 + (float)c * (colw + gp);
		for (int r = 0; r < n; r++) {
			const float t = (float)r / (float)n;
			if (t > 0.5f && (n - 1 - r) % 2 == 1)
				continue;
			const uint32_t col = ew_color(e, (t - 0.3f) / 0.7f, am, linear);
			ew_rect(e, x, mid - (float)(r + 1) * px, colw, px, col);
			ew_rect(e, x, mid + (float)r * px, colw, px, col);
		}
	}
}

static void ew_draw_pixwave(struct ew *e, float W, float H, bool linear)
{
	const int n = ew_ghosts(e, 4);
	for (int age = n; age >= 1; age--) {
		ew_use_wave(e, age);
		ew_pixwave_pass(e, W, H, linear, ew_ghost_alpha(age, n) * 0.7f);
	}
	ew_use_wave(e, 0);
	ew_pixwave_pass(e, W, H, linear, 1.0f);
}

static const uint8_t ew_bayer[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};

/* ordered dither: is the pixel at (cx, cy) lit for a density of d (0..1)? */
static bool ew_dither(int cx, int cy, float d)
{
	return d > ((float)ew_bayer[cy & 3][cx & 3] + 0.5f) / 16.0f;
}

/* bars filled with a dither pattern: dense at the base, breaking up into scattered pixels toward the top */
static void ew_draw_dither(struct ew *e, float W, float H, bool linear)
{
	const float px = (float)e->px;
	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	int rows = (int)floorf(H / px);
	if (rows < 4)
		rows = 4;
	const float gp = floorf((float)e->gap / px) * px;
	const uint32_t dash = ew_color(e, 1.0f, 1.0f, linear);
	const uint32_t ghost = ew_color(e, 1.0f, 0.6f, linear);

	for (int j = 0; j < n2; j++) {
		const int i = !e->mirror ? j : (j < nb ? nb - 1 - j : j - nb);
		const float x = ew_snap((float)j * W / (float)n2, px);
		const float w = ew_snap((float)(j + 1) * W / (float)n2, px) - x - gp;
		if (w <= 0.0f)
			continue;
		const int cols = (int)roundf(w / px);
		const int cx0 = (int)roundf(x / px);

		int n = (int)roundf(e->val[i] * (float)(rows - 2));
		if (n < 1)
			n = e->baseline ? 1 : 0;
		int top = n;
		if (e->trails > 0.0f) {
			const int tr = (int)roundf(e->trail[i] * (float)(rows - 2));
			if (tr > top)
				top = tr;
		}

		for (int r = 0; r < top; r++) {
			const float t = n > 0 ? (float)r / (float)n : 1.0f;
			const float dens = r < n ? 1.0f - powf(t, 1.4f) * 0.92f : 0.2f;
			const uint32_t c = r < n ? ew_color(e, t, 1.0f, linear) : ghost;
			const float y = H - (float)(r + 1) * px;
			int run = -1;
			for (int k = 0; k <= cols; k++) {
				const bool lit = k < cols && ew_dither(cx0 + k, rows - 1 - r, dens);
				if (lit && run < 0)
					run = k;
				if (!lit && run >= 0) {
					ew_rect(e, x + (float)run * px, y, (float)(k - run) * px, px, c);
					run = -1;
				}
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

/* ---- sea styles, ported from the zig~zag "earth" map skin ---- */

static float ew_clamp01(float x)
{
	return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

/* how visible the always-moving styles are: they fade out at silence unless "Show Baseline When Silent" is on */
static float ew_presence(const struct ew *e)
{
	return e->baseline ? 1.0f : ew_clamp01(e->energy * 6.0f);
}

/* the ocean: a sea surface that rises with the spectrum, made of ~, the double wave and shade blocks */
static void ew_draw_ocean(struct ew *e, float W, float H, bool linear)
{
	float u = roundf((float)e->font / 8.0f);
	if (u < 1.0f)
		u = 1.0f;
	const float cw = u * 6.0f, lh = u * 8.0f;
	const int cols = (int)floorf(W / cw);
	const int rows = (int)floorf(H / lh);
	const float vis = ew_presence(e);
	if (cols < 2 || rows < 2 || vis < 0.01f)
		return;
	const int n2 = e->mirror ? e->band_n * 2 : e->band_n;
	const float tt = (float)e->clock;
	const float hb = (float)rows * lh;
	float bass = ew_band_avg(e, 0.0f, 0.2f);
	if (bass > 1.0f)
		bass = 1.0f;
	const float pump = ew_clamp01(0.8f + e->beat * 0.35f);
	const uint32_t under = ew_color(e, 1.0f, 0.3f * vis, linear);

	for (int c = 0; c < cols; c++) {
		const float xs = ((float)c + 0.5f) / (float)cols;
		const float v = ew_spec_smooth(e, e->val, xs * (float)(n2 - 1), n2);
		const float surf = (hb * (0.22f + 0.65f * powf(v, 0.8f)) +
				    lh * 3.5f * sinf(xs * 9.0f + tt * 1.4f) * (0.4f + bass * 1.4f) * 0.35f +
				    lh * 1.4f * sinf(xs * 27.0f - tt * 2.6f) * 0.35f + e->beat * lh * 0.7f) *
				   vis;
		for (int rb = 0; rb < rows; rb++) {
			const float depth = surf / lh - (float)rb;
			if (depth < 0.0f)
				break;
			const int g = depth < 1.0f   ? G_TILDE
				      : depth < 2.0f ? G_APPROX
				      : depth < 4.0f ? G_SHADE1
						     : G_SHADE2;
			const float x = (float)c * cw, y = H - (float)(rb + 1) * lh;
			if (depth >= 1.0f)
				ew_rect(e, x, y, cw, lh, under);
			ew_glyph(e, g, x, y, u, ew_color(e, depth / 7.0f, pump * vis, linear));
		}
	}
}

/* 32 glyphs, 16x16, rows top to bottom, bit 15 = left column:
   "・ーノくしつのミラネ木水音波夢響0123456789:<>=+*"
   rasterised from Noto Sans CJK JP Bold (SIL Open Font License 1.1) */
#define KG_COUNT 32
#define KG_KANA 16 /* the first 16 run light -> dense */
#define KG_DIGITS 16
static const uint16_t ew_kana_glyphs[KG_COUNT][16] = {
	{0x0000, 0x0000, 0x0000, 0x0000, 0x01C0, 0x01C0, 0x01C0, 0x01C0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x3FFE, 0x3FFE, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0018, 0x0018, 0x0038, 0x0030, 0x0070, 0x0060, 0x00E0, 0x01C0, 0x0380, 0x0F00, 0x1E00, 0x0800,
	 0x0000, 0x0000},
	{0x0000, 0x0020, 0x0070, 0x00E0, 0x03C0, 0x0700, 0x0E00, 0x0C00, 0x0E00, 0x0700, 0x03C0, 0x01E0, 0x00F0, 0x0070,
	 0x0000, 0x0000},
	{0x0000, 0x0E00, 0x0C00, 0x0C00, 0x0C00, 0x0C00, 0x0C00, 0x0C00, 0x0C00, 0x0C00, 0x0C04, 0x0C0E, 0x0E3C, 0x07F8,
	 0x03E0, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x1FFC, 0x3E0C, 0x300E, 0x0006, 0x0006, 0x000E, 0x001C, 0x0078, 0x03F0, 0x03C0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x03F0, 0x0FF8, 0x1DDC, 0x198C, 0x3186, 0x3186, 0x3186, 0x330E, 0x3F0C, 0x1E1C, 0x0CF8, 0x0070,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x07E0, 0x07F8, 0x0038, 0x0000, 0x0F80, 0x0FF0, 0x0070, 0x0000, 0x0E00, 0x1FE0, 0x03F8, 0x0018,
	 0x0000, 0x0000},
	{0x0000, 0x0FF8, 0x0FF8, 0x0000, 0x0000, 0x1FFC, 0x1FFC, 0x000C, 0x001C, 0x0018, 0x0070, 0x00F0, 0x03E0, 0x0380,
	 0x0000, 0x0000},
	{0x0000, 0x01C0, 0x00C0, 0x1FF8, 0x1FFC, 0x0038, 0x0070, 0x01E0, 0x07D8, 0x1FDC, 0x3CCE, 0x10C0, 0x00C0, 0x01C0,
	 0x0080, 0x0000},
	{0x01C0, 0x01C0, 0x01C0, 0x01C0, 0x3FFE, 0x3FFE, 0x03E0, 0x03E0, 0x07F0, 0x0FD8, 0x1DDC, 0x39CE, 0x31C6, 0x01C0,
	 0x01C0, 0x0000},
	{0x0000, 0x00C0, 0x00C0, 0x00C6, 0x3FCE, 0x3FFC, 0x06F8, 0x0EF0, 0x0CF0, 0x0CF8, 0x18DC, 0x38CE, 0x31CE, 0x0782,
	 0x0780, 0x0000},
	{0x0180, 0x1FFC, 0x1FFE, 0x0638, 0x0630, 0x3FFE, 0x3FFE, 0x0FF8, 0x0FF8, 0x0C18, 0x0C18, 0x0FF8, 0x0C18, 0x0FF8,
	 0x0FF8, 0x0000},
	{0x1030, 0x1C30, 0x0BFE, 0x03FE, 0x2336, 0x3B34, 0x1BFC, 0x03FC, 0x03CC, 0x0B7C, 0x1B78, 0x1A38, 0x367C, 0x37EE,
	 0x0486, 0x0000},
	{0x3FFE, 0x3FFE, 0x0630, 0x1FFC, 0x1B6C, 0x1B6C, 0x1FFC, 0x0000, 0x3FFE, 0x3306, 0x3FF6, 0x1E70, 0x03E0, 0x1F80,
	 0x1E00, 0x0000},
	{0x1BE0, 0x3F7E, 0x3BFE, 0x1F7E, 0x3FFA, 0x3F5A, 0x1BFE, 0x3FFC, 0x0630, 0x0630, 0x3FFE, 0x0FF8, 0x0FFC, 0x0C0C,
	 0x0FFC, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x03E0, 0x0670, 0x0630, 0x0630, 0x0630, 0x0630, 0x0630, 0x0670, 0x03E0, 0x01C0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x03C0, 0x01C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x00C0, 0x07F0, 0x07F0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x07E0, 0x0460, 0x0070, 0x0060, 0x0060, 0x00C0, 0x01C0, 0x0380, 0x07F0, 0x07F0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x07E0, 0x0070, 0x0060, 0x01E0, 0x01C0, 0x0070, 0x0030, 0x0470, 0x07E0, 0x03C0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x01E0, 0x01E0, 0x0360, 0x0360, 0x0660, 0x0FF0, 0x0FF0, 0x0060, 0x0060, 0x0060, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x07F0, 0x0600, 0x0600, 0x07E0, 0x07F0, 0x0070, 0x0030, 0x0470, 0x07E0, 0x03C0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x03F0, 0x0700, 0x0600, 0x06E0, 0x07F0, 0x0630, 0x0630, 0x0630, 0x03F0, 0x01E0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x07F0, 0x0060, 0x0060, 0x00C0, 0x00C0, 0x01C0, 0x0180, 0x0180, 0x0180, 0x0180, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x07E0, 0x0630, 0x0630, 0x03E0, 0x03E0, 0x0670, 0x0630, 0x0630, 0x07F0, 0x03E0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x07E0, 0x0670, 0x0630, 0x0630, 0x07F0, 0x03B0, 0x0030, 0x0060, 0x07E0, 0x03C0, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x0000, 0x01C0, 0x0180, 0x0000, 0x0000, 0x0000, 0x0080, 0x01C0, 0x0180, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x0000, 0x0070, 0x01F0, 0x0780, 0x0700, 0x03C0, 0x00F0, 0x0010, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x0000, 0x0600, 0x07C0, 0x00F0, 0x0070, 0x03E0, 0x0780, 0x0400, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x0000, 0x07F0, 0x0000, 0x0000, 0x07F0, 0x07F0, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0000, 0x0080, 0x0080, 0x0080, 0x07F0, 0x07F0, 0x0080, 0x0080, 0x0080, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
	{0x0000, 0x0000, 0x0100, 0x0180, 0x07E0, 0x0380, 0x03C0, 0x0240, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
	 0x0000, 0x0000},
};

/* stable pseudo-random number 0..1 from three integers */
static float ew_rand3(int a, int b, int c)
{
	uint32_t h = (uint32_t)a * 0x8DA6B343u ^ (uint32_t)b * 0xD8163841u ^ (uint32_t)c * 0xCB1AB31Fu;
	h ^= h >> 15;
	h *= 0x2C1B3C6Du;
	h ^= h >> 12;
	h *= 0x297A2D39u;
	h ^= h >> 15;
	return (float)(h >> 8) / 16777216.0f;
}

/* one 16x16 glyph from the atlas, drawn as runs of pixels so each row is only a few rectangles */
static void ew_kglyph(struct ew *e, int g, float x, float y, float u, uint32_t c)
{
	for (int r = 0; r < 16; r++) {
		const uint16_t bits = ew_kana_glyphs[g][r];
		int run = -1;
		for (int col = 0; col <= 16; col++) {
			const bool on = col < 16 && (bits & (0x8000 >> col));
			if (on && run < 0)
				run = col;
			if (!on && run >= 0) {
				ew_rect(e, x + (float)run * u, y + (float)r * u, (float)(col - run) * u, u, c);
				run = -1;
			}
		}
	}
}

/* spectrum level for column c of cols (follows Mirror and the band count) */
static float ew_col_level(const struct ew *e, int c, int cols)
{
	const int n2 = e->mirror ? e->band_n * 2 : e->band_n;
	const float p = ((float)c + 0.5f) / (float)cols * (float)n2;
	int j = (int)floorf(p);
	if (j > n2 - 1)
		j = n2 - 1;
	const int i = !e->mirror ? j : (j < e->band_n ? e->band_n - 1 - j : j - e->band_n);
	return e->val[i];
}

/* layered bars standing on a waterline, with a fainter reflection underneath that ripples */
static void ew_draw_reflect(struct ew *e, float W, float H, bool linear)
{
	const float px = (float)e->px;
	const float line = ew_snap(H * 0.64f, px);
	ew_draw_bars(e, W, line, linear);

	const int nb = e->band_n;
	const int n2 = e->mirror ? nb * 2 : nb;
	int rows = (int)floorf(line / px);
	if (rows < 4)
		rows = 4;
	const float gp = floorf((float)e->gap / px) * px;
	const float tt = (float)e->clock;
	const int max_r = (int)floorf((H - line - px) / px);

	for (int j = 0; j < n2; j++) {
		const int i = !e->mirror ? j : (j < nb ? nb - 1 - j : j - nb);
		const float x = ew_snap((float)j * W / (float)n2, px);
		const float w = ew_snap((float)(j + 1) * W / (float)n2, px) - x - gp;
		if (w <= 0.0f)
			continue;
		int n = (int)roundf(e->val[i] * (float)(rows - 2));
		if (n < 1)
			n = e->baseline ? 1 : 0;
		int depth = (int)roundf((float)n * 0.6f);
		if (depth > max_r)
			depth = max_r;
		for (int r = 0; r < depth; r++) {
			if (r % 2 == 1)
				continue;
			const float f = (float)r / (float)(depth > 1 ? depth : 1);
			const float shift =
				roundf(sinf((float)r * 0.9f - tt * 6.0f + (float)j * 0.35f) * (1.0f + f * 2.0f)) * px *
				0.5f;
			const float t = (float)r / (float)(n > 1 ? n : 1);
			ew_rect(e, x + shift, line + px + (float)r * px, w, px,
				ew_color(e, t, 0.5f * (1.0f - f), linear));
		}
	}
}

/* a row of buildings: heights follow the music, windows light up as it gets louder */
static void ew_draw_skyline(struct ew *e, float W, float H, bool linear)
{
	const float px = (float)e->px;
	const float vis = ew_presence(e);
	if (vis < 0.01f)
		return;
	const float base_w = px * 12.0f;
	const uint32_t edge = ew_color(e, 0.0f, vis, linear);
	const uint32_t body = ew_color(e, 1.0f, 0.14f * vis, linear);
	const int slot = (int)floorf((float)e->clock * 1.4f); /* windows flicker on and off a little */
	float x = 0.0f;

	for (int b = 0; x < W; b++) {
		static const float widths[4] = {0.75f, 1.0f, 1.3f, 1.65f};
		float bw = ew_snap(base_w * widths[(int)(ew_rand3(b, 1, 7) * 4.0f) & 3], px);
		if (bw < px * 4.0f)
			bw = px * 4.0f;
		if (x + bw > W)
			bw = W - x;
		const int cols = (int)floorf(W / bw) > 0 ? (int)floorf(W / bw) : 1;
		const int c_mid = (int)((x + bw * 0.5f) / W * (float)cols);
		const float v = ew_col_level(e, c_mid, cols);
		const float h = ew_snap((0.12f + 0.85f * v) * (H - px * 6.0f) * vis, px);
		if (h >= px * 3.0f) {
			const float top = H - h;
			const float x0 = x, x1 = x + bw - px; /* one pixel of street between buildings */
			ew_rect(e, x0, top, x1 - x0, h, body);
			ew_rect(e, x0, top, x1 - x0, px, edge); /* roof */
			ew_rect(e, x0, top, px, h, edge);       /* walls */
			ew_rect(e, x1 - px, top, px, h, edge);
			if (ew_rand3(b, 2, 9) < 0.35f) /* antenna */
				ew_rect(e, ew_snap((x0 + x1) * 0.5f, px), top - px * 4.0f, px, px * 4.0f, edge);

			/* windows: px wide, px+px tall, on a 2px x 3px grid, lit from the ground up */
			const float wx0 = x0 + px * 2.0f, wy0 = top + px * 2.0f;
			const int wcols = (int)floorf((x1 - px * 2.0f - wx0) / (px * 2.0f));
			const int wrows = (int)floorf((H - px - wy0) / (px * 3.0f));
			for (int wr = 0; wr < wrows; wr++) {
				const float from_ground = 1.0f - (float)wr / (float)(wrows > 1 ? wrows : 1);
				for (int wc = 0; wc < wcols; wc++) {
					const float k =
						ew_rand3(b * 131 + wc, wr, slot + (int)(ew_rand3(b, wc, wr) * 3.0f));
					if (k < v * 1.25f - (1.0f - from_ground) * 0.35f)
						ew_rect(e, wx0 + (float)wc * px * 2.0f, wy0 + (float)wr * px * 3.0f, px,
							px * 2.0f,
							ew_color(e, ew_rand3(b, wc, wr + 50) * 0.8f, vis, linear));
				}
			}
		}
		x += bw;
	}
}

/* spectrum columns of Japanese characters: light kana at the tips, dense kanji lower down (from zig~zag) */
static void ew_draw_kana(struct ew *e, float W, float H, bool linear)
{
	float u = roundf((float)e->font / 16.0f);
	if (u < 1.0f)
		u = 1.0f;
	const float cell = u * 17.0f;
	const int cols = (int)floorf(W / cell);
	const int rows = (int)floorf(H / cell);
	const float vis = ew_presence(e);
	if (cols < 1 || rows < 1 || vis < 0.01f)
		return;
	const float x_off = floorf((W - (float)cols * cell) * 0.5f);
	const int slot = (int)floorf((float)e->clock / 0.11f);

	for (int c = 0; c < cols; c++) {
		const float v = ew_col_level(e, c, cols);
		const float h = powf(v, 1.3f) * (float)rows * 1.1f * (1.0f + e->beat * 0.2f) * vis;
		for (int rb = 0; rb < rows && (float)rb < h; rb++) {
			const float from_top = h - (float)rb;
			const float dens = ew_clamp01(from_top / 7.0f) * (0.4f + 0.6f * v);
			int g = (int)floorf(dens * 15.0f);
			const float flick = ew_rand3(c, rb, slot);
			g += flick > 0.7f ? 1 : (flick < 0.3f ? -1 : 0);
			if (g < 0)
				g = 0;
			if (g > KG_KANA - 1)
				g = KG_KANA - 1;
			const float t = (float)rb / (float)(rows > 1 ? rows - 1 : 1);
			const uint32_t col = from_top < 1.0f ? ew_color(e, 0.0f, vis, linear)
							     : ew_color(e, t, (0.55f + 0.45f * dens) * vis, linear);
			ew_kglyph(e, g, x_off + (float)c * cell, H - (float)(rb + 1) * cell, u, col);
		}
	}
}

/* Matrix rain: drops of katakana and digits, thicker and faster when the music is louder */
static void ew_draw_rain(struct ew *e, float W, float H, bool linear)
{
	float u = roundf((float)e->font / 16.0f);
	if (u < 1.0f)
		u = 1.0f;
	const float cell = u * 17.0f;
	const int cols = (int)floorf(W / cell);
	const int rows = (int)floorf(H / cell);
	const float vis = ew_presence(e);
	if (cols < 1 || rows < 1 || vis < 0.01f)
		return;
	const float x_off = floorf((W - (float)cols * cell) * 0.5f);
	const float top = H - (float)rows * cell;
	static const int pool[] = {2, 4, 5, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
	const int pool_n = (int)(sizeof(pool) / sizeof(pool[0]));

	for (int c = 0; c < cols; c++) {
		const float v = ew_col_level(e, c, cols);
		for (int k = 0; k < 2; k++) {
			const float rnd = ew_rand3(c, k, 3);
			const float sp = 0.6f + rnd * 0.9f;
			const int len = 4 + (int)(ew_rand3(c, k, 4) * 8.0f);
			const float span = (float)(rows + len);
			const float pos = e->rain_t * sp * 6.0f + rnd * span * 3.0f;
			const int drop = (int)floorf(pos / span);
			const float head = fmodf(pos, span);
			/* each drop only falls if its column is loud enough this time round */
			const float need = ew_rand3(c, k * 977 + drop, 5) * 0.75f;
			const float show = ew_clamp01((v - need) * 5.0f) * vis;
			if (show < 0.02f)
				continue;
			const int gslot = (int)floorf(e->rain_t * sp * 2.0f);
			for (int q = 0; q < len; q++) {
				const int r = (int)floorf(head) - q;
				if (r < 0 || r >= rows)
					continue;
				const float fade = 1.0f - (float)q / (float)len;
				const int g = pool[(int)(ew_rand3(c * 7 + k, r, gslot + q) * (float)pool_n) % pool_n];
				const uint32_t col =
					q == 0 ? ew_color(e, 0.0f, show, linear)
					       : ew_color(e, 0.25f + 0.75f * (1.0f - fade), fade * show, linear);
				ew_kglyph(e, g, x_off + (float)c * cell, top + (float)r * cell, u, col);
			}
		}
	}
}

static void ew_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct ew *e = data;
	float W = (float)e->width;
	const float H = (float)e->height;
	const bool linear = gs_get_linear_srgb();

	if (e->band_n <= 0 || W < 1.0f || H < 1.0f)
		return;

	e->vn = 0;
	e->ox = 0.0f;
	if (e->bg_on)
		ew_rect(e, 0.0f, 0.0f, W, H, ew_plain(e->bg, linear));

	/* Length: draw the same visual into a narrower strip centred in the source */
	const float full_w = W;
	W = roundf(full_w * e->length);
	if (W < 8.0f)
		W = 8.0f;
	e->ox = floorf((full_w - W) * 0.5f);

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
	case STYLE_LED:
		ew_draw_led(e, W, H, linear);
		break;
	case STYLE_RIDGE:
		ew_draw_ridge(e, W, H, linear);
		break;
	case STYLE_PIXWAVE:
		ew_draw_pixwave(e, W, H, linear);
		break;
	case STYLE_CURVE:
		ew_draw_curve(e, W, H, linear);
		break;
	case STYLE_DITHER:
		ew_draw_dither(e, W, H, linear);
		break;
	case STYLE_OCEAN:
		ew_draw_ocean(e, W, H, linear);
		break;
	case STYLE_REFLECT:
		ew_draw_reflect(e, W, H, linear);
		break;
	case STYLE_SKYLINE:
		ew_draw_skyline(e, W, H, linear);
		break;
	case STYLE_KANA:
		ew_draw_kana(e, W, H, linear);
		break;
	case STYLE_RAIN:
		ew_draw_rain(e, W, H, linear);
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
	const bool bars = style == STYLE_BARS || style == STYLE_SOLID || style == STYLE_CENTER || style == STYLE_LED ||
			  style == STYLE_DITHER;
	const bool spectrum = bars || style == STYLE_RIDGE || style == STYLE_CURVE || style == STYLE_OCEAN ||
			      style == STYLE_REFLECT || style == STYLE_SKYLINE || style == STYLE_KANA ||
			      style == STYLE_RAIN;
	const bool grad_ok = style != STYLE_CENTER && style != STYLE_ASCII;
	const bool grad = obs_data_get_bool(settings, "gradient");

	ew_show(props, "mirror", spectrum && style != STYLE_CENTER);
	ew_show(props, "range", spectrum);
	ew_show(props, "gradient", grad_ok);
	ew_show(props, "color2", grad && grad_ok);
	ew_show(props, "gradient3", grad && grad_ok);
	ew_show(props, "color3", grad && grad_ok && obs_data_get_bool(settings, "gradient3"));
	ew_show(props, "bg", obs_data_get_bool(settings, "bg_on"));
	ew_show(props, "scale", spectrum);
	ew_show(props, "smooth", spectrum || style == STYLE_SINES);
	ew_show(props, "decay", spectrum || style == STYLE_SINES);
	ew_show(props, "bars", spectrum);
	ew_show(props, "gap", bars || style == STYLE_REFLECT);
	ew_show(props, "px",
		style == STYLE_BARS || style == STYLE_LED || style == STYLE_PIXWAVE || style == STYLE_DITHER ||
			style == STYLE_REFLECT || style == STYLE_SKYLINE);
	ew_show(props, "peaks",
		style == STYLE_BARS || style == STYLE_SOLID || style == STYLE_LED || style == STYLE_DITHER ||
			style == STYLE_REFLECT);
	ew_show(props, "baseline",
		bars || style == STYLE_OCEAN || style == STYLE_REFLECT || style == STYLE_SKYLINE ||
			style == STYLE_KANA || style == STYLE_RAIN);
	ew_show(props, "layers", style == STYLE_SCOPE || style == STYLE_RIDGE);
	ew_show(props, "font",
		style == STYLE_ASCII || style == STYLE_OCEAN || style == STYLE_KANA || style == STYLE_RAIN);
	ew_show(props, "sine_count", style == STYLE_SINES);
	ew_show(props, "sine_speed", style == STYLE_SINES);
	ew_show(props, "sine_thick", style == STYLE_SINES);
	ew_show(props, "sine_size", style == STYLE_SINES);
	ew_show(props, "sine_react", style == STYLE_SINES);
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
	obs_property_list_add_int(p, "LED meter", STYLE_LED);
	obs_property_list_add_int(p, "Ridgeline", STYLE_RIDGE);
	obs_property_list_add_int(p, "Pixel waveform", STYLE_PIXWAVE);
	obs_property_list_add_int(p, "Spectrum curve", STYLE_CURVE);
	obs_property_list_add_int(p, "Bars (dither)", STYLE_DITHER);
	obs_property_list_add_int(p, "Ocean (zig~zag)", STYLE_OCEAN);
	obs_property_list_add_int(p, "Reflection", STYLE_REFLECT);
	obs_property_list_add_int(p, "Skyline", STYLE_SKYLINE);
	obs_property_list_add_int(p, "Kana columns (zig~zag)", STYLE_KANA);
	obs_property_list_add_int(p, "Matrix rain", STYLE_RAIN);
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
	obs_properties_add_float_slider(props, "length", "Length", 0.1, 1.0, 0.01);

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
	obs_properties_add_int_slider(props, "font", "Character Size", 8, 48, 1);

	obs_properties_add_int_slider(props, "sine_count", "Waves", 1, 12, 1);
	obs_properties_add_float_slider(props, "sine_speed", "Wave Speed", 0.0, 4.0, 0.05);
	obs_properties_add_int_slider(props, "sine_thick", "Wave Thickness", 1, 8, 1);
	obs_properties_add_float_slider(props, "sine_size", "Wave Size", 0.2, 3.0, 0.05);
	obs_properties_add_float_slider(props, "sine_react", "Wave Reactivity", 0.0, 3.0, 0.05);

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
	obs_data_set_default_double(s, "sine_react", 1.5);
	obs_data_set_default_double(s, "length", 1.0);
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
	e->style = ew_clampi(obs_data_get_int(s, "style"), STYLE_BARS, STYLE_LAST);
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
	e->sine_react = ew_clampf(obs_data_get_double(s, "sine_react"), 0.0f, 3.0f);
	e->length = ew_clampf(obs_data_get_double(s, "length"), 0.1f, 1.0f);
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
