/*
 * This file is part of vitaGL
 * Copyright 2017, 2018, 2019, 2020 Rinnegatamante
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/* 
 * gxm_utils.c:
 * Utilities for GXM api usage
 */
#include "../shared.h"

// flat ring, no GPU fence on wrap: must hold more than (frames-in-flight + 1)
// frames of uniforms or the wrap overwrites data the GPU is still reading
// (wrong-color flashes, worse under MSAA where the GPU lags further).
// [MP4-TEXFIX test 2026-07-24] 8MB starved the VGL_MEM_RAM pool -> texture allocs
// failed -> white untextured surfaces (party-mode chars/bg). Revert to 2MB to confirm.
// Every draw's vertex AND fragment default uniform buffer is carved from this
// pool (vglReserveVertexUniformBuffer / vglReserveFragmentUniformBuffer), and
// the wrap below has NO GPU FENCE -- it just rewinds to the start. So the pool
// must be larger than the uniforms consumed by every frame the GPU still has
// in flight, or a wrap overwrites uniforms a queued draw has not read yet.
//
// Open Nectar hit exactly that: ~718 draws x ~2.7KB = ~2MB per frame wrapped
// this 2MB pool about once per frame, silently (the outage warning below only
// fires if it wraps TWICE in one frame), corrupting whichever tiles the GPU
// rasterised after the wrap -- garbage pinned to one side of the screen and
// different every frame. Raising it is the fix; sceGxmFinish per frame also
// cured it but cost half the frame rate.
//
// 16MB covers ~8 frames at the title and ~4 in heavier scenes, comfortably
// past displayQueueMaxPendingCount (= display buffers - 1).
#define UNIFORM_CIRCULAR_POOL_SIZE (16 * 1024 * 1024)

void *vgl_def_frag_buf = NULL;
void *vgl_def_vert_buf = NULL;
static uint8_t *unif_pool = NULL;
static uint32_t unif_idx = 0;
// Running total of the bytes handed out (vglGetFastPathStats): the per-frame
// consumption says how close a frame comes to lapping the fenceless ring.
uint32_t vgl_unif_ring_bytes = 0;

void vglSetupUniformCircularPool() {
	unif_pool = gpu_alloc_mapped(UNIFORM_CIRCULAR_POOL_SIZE, VGL_MEM_RAM);
}

void *vglReserveUniformCircularPoolBuffer(uint32_t size) {
	void *r;
	vgl_unif_ring_bytes += size;
	if (unif_idx + size >= UNIFORM_CIRCULAR_POOL_SIZE) {
#ifndef SKIP_ERROR_HANDLING
		static uint32_t last_frame_swap = 0;
		if (last_frame_swap == vgl_framecount) {
			vgl_log("%s:%d Circular Uniform Pool outage detected! Consider increasing its size...\n", __FILE__, __LINE__);
		}
		last_frame_swap = vgl_framecount;
#endif
		r = unif_pool;
		unif_idx = size;
	} else {
		r = (unif_pool + unif_idx);
		unif_idx += size;
	}
	return r;
}
