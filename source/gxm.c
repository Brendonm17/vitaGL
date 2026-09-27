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
 * gxm.c:
 * Implementation for setup and cleanup for sceGxm specific stuffs
 */

#include "shared.h"

#define MAX_SCENES_PER_FRAME 8 // Maximum amount of scenes per frame allowed by sceGxm per render target

// FIXME: Since we use our own default uniform buffers circular pool, fragment and vertex buffer rings can likely be reduced in size
uint32_t gxm_param_buf_size = SCE_GXM_DEFAULT_PARAMETER_BUFFER_SIZE; // Param buffer size for sceGxm
static uint32_t gxm_vdm_buf_size = SCE_GXM_DEFAULT_VDM_RING_BUFFER_SIZE; // VDM ring buffer size for sceGxm
static uint32_t gxm_vertex_buf_size = SCE_GXM_DEFAULT_VERTEX_RING_BUFFER_SIZE; // Vertex ring buffer size for sceGxm
static uint32_t gxm_fragment_buf_size = SCE_GXM_DEFAULT_FRAGMENT_RING_BUFFER_SIZE; // Fragment ring buffer size for sceGxm
static uint32_t gxm_usse_buf_size = SCE_GXM_DEFAULT_FRAGMENT_USSE_RING_BUFFER_SIZE; // Fragment ring buffer size for sceGxm

// Shader Patcher memory configuration
static unsigned int shader_patcher_buffer_size = 1024 * 1024;
static unsigned int shader_patcher_vertex_usse_size = 1024 * 1024;
static unsigned int shader_patcher_fragment_usse_size = 1024 * 1024;

static void *vdm_ring_buffer_addr; // VDM ring buffer memblock starting address
static void *vertex_ring_buffer_addr; // vertex ring buffer memblock starting address
static void *fragment_ring_buffer_addr; // fragment ring buffer memblock starting address
static void *fragment_usse_ring_buffer_addr; // fragment USSE ring buffer memblock starting address

static SceGxmRenderTarget *gxm_render_target; // Display render target
SceGxmColorSurface gxm_color_surfaces[DISPLAY_MAX_BUFFER_COUNT]; // Display color surfaces
uint8_t gxm_display_buffer_count = 3; // Display buffers count
void *gxm_color_surfaces_addr[DISPLAY_MAX_BUFFER_COUNT]; // Display color surfaces memblock starting addresses
static SceGxmSyncObject *gxm_sync_objects[DISPLAY_MAX_BUFFER_COUNT]; // Display sync objects
unsigned int gxm_front_buffer_index; // Display front buffer id
unsigned int gxm_back_buffer_index; // Display back buffer id
static void (*vgl_display_cb)(void *framebuf) = NULL; // Additional custom callback used inside display queue callback

static void *gxm_shader_patcher_buffer_addr; // Shader PAtcher buffer memblock starting address
static void *gxm_shader_patcher_vertex_usse_addr; // Shader Patcher vertex USSE memblock starting address
static void *gxm_shader_patcher_fragment_usse_addr; // Shader Patcher fragment USSE memblock starting address

SceGxmDepthStencilSurface gxm_depth_stencil_surface; // Depth/Stencil surfaces setup for sceGxm

static SceUID shared_fb; // In-use hared framebuffer identifier
static SceSharedFbInfo shared_fb_info; // In-use shared framebuffer info struct
framebuffer *in_use_framebuffer = NULL; // Currently in use framebuffer
framebuffer *old_framebuffer = NULL; // Framebuffer used in last scene
GLboolean dirty_framebuffer = GL_FALSE; // Flag whether current in use framebuffer is invalidated
GLboolean dirty_query = GL_FALSE; // Flag whether occlusion queries needs results
static GLboolean needs_end_scene = GL_FALSE; // Flag for gxm end scene requirement at scene reset
static GLboolean needs_scene_reset = GL_TRUE; // Flag for when a scene reset is required
static GLboolean display_depth_persist = GL_FALSE; // Flag for display depth/stencil persistence across vglFlushFrame scene splits (see vglSetDisplayDepthPersistence)
static GLboolean display_depth_load_pending = GL_FALSE; // One-shot: the next display scene must force-load the stored depth (armed by vglFlushFrame, consumed in sceneReset)

// [EFB-ASYNC v3] capture-transfer fences + instrumentation (see vglCaptureFramebufferRegion).
// capture_tn = the transfer-done notification: notification-region slot 1 (slot 0 is query_fence,
// buffers.c); value is a monotonic per-transfer counter the transfer engine writes back on completion.
static SceGxmNotification capture_tn = { NULL, 0 };
static GLboolean capture_tn_pending = GL_FALSE; // an enqueued capture transfer has not been TN-waited yet
// [FBORT] one-shot force-LOAD arming for the FBO depth restart after vglCaptureFboRegion's scene split
// (consumed at the next FBO sceneReset BeginScene, mirroring display_depth_load_pending). fbo_depth_load_fb
// pins WHICH FBO surface was armed so the disarm targets the right depthbuffer_ptr.
static GLboolean fbo_depth_load_pending = GL_FALSE;
static framebuffer *fbo_depth_load_fb = NULL;
// Exported wait meters (vitaGL.h): FN = pre-transfer fragment-fence wait (vglCaptureFramebufferRegion),
// TN = pre-EndScene transfer-fence wait (sceneEnd). Microseconds; the app prints/alerts on these.
uint32_t vgl_efbwait_fn_us = 0, vgl_efbwait_fn_max = 0, vgl_efbwait_fn_cnt = 0;
uint32_t vgl_efbwait_tn_us = 0, vgl_efbwait_tn_max = 0, vgl_efbwait_tn_cnt = 0;
// [EFB-ASYNC v3.1] FN-wait BACKLOG histogram: how many un-retired scenes the GPU held when the capture's
// FN wait started (query_fence.value - *query_fence.address; the GPU writes each retired scene's value
// back to address). Device motive (20fps cards scene 2026-07-06: fn avg ~47ms/copy at 38 draws/1.7K
// verts = GPU backlog, but fn alone can't say WHICH scene is the hog): bk[1] = only the just-submitted
// pre-capture scene was pending (its own fill is the cost); bk[2+] = the PREVIOUS frame's post-capture
// composite scene had not retired either (the LINEAR-sampled capture composite is the suspect). Clamped.
uint32_t vgl_efbwait_bk[4] = { 0, 0, 0, 0 };
// [EFB-ASYNC v3.2] FN-wait retirement SPLIT (device bk=0/1/703/0: two scenes pending at ~every capture,
// so the ~47ms wait is prev-scene tail + own-scene exec, but which?). The wait is a 100us poll on the
// notification address; the tick where it advances to value-1 splits the wait: prev = time until every
// OLDER scene (the previous frame's post-capture composite) retired, own = the remainder (the just-
// submitted pre-capture scene's own execution). prev-dominant -> the visible composite scene is the GPU
// hog; own-dominant -> the shadow/pre-capture scene is. Poll granularity (~100us) is noise at these
// magnitudes.
uint32_t vgl_efbwait_prev_us = 0, vgl_efbwait_own_us = 0;
// Present-path meters (vglGetPresentStats, vitaGL.h): running microsecond totals around every call a
// present can block in, read by the app instead of logged. The display queue blocks while all display
// buffers are pending, EndScene/BeginScene block on the render target's scene resources and the colour
// buffer's display sync object, and the GC handshake blocks if the collector thread lags a purge behind.
static uint32_t vgl_ps_swaps = 0;
static uint32_t vgl_ps_end_scene_us = 0, vgl_ps_end_scenes = 0;
static uint32_t vgl_ps_display_queue_us = 0, vgl_ps_gc_wait_us = 0;
static uint32_t vgl_ps_first_begin_us = 0, vgl_ps_begin_us = 0, vgl_ps_begins = 0;
static GLboolean vgl_ps_first_begin_pending = GL_TRUE; // the next sceGxmBeginScene is the frame's first
static inline __attribute__((always_inline)) void vgl_ps_note_begin(uint32_t t0) {
	const uint32_t dt = sceKernelGetProcessTimeLow() - t0;
	if (vgl_ps_first_begin_pending) {
		vgl_ps_first_begin_pending = GL_FALSE;
		vgl_ps_first_begin_us += dt;
	} else {
		vgl_ps_begin_us += dt;
		vgl_ps_begins++;
	}
}

SceGxmContext *gxm_context; // sceGxm context instance
GLenum vgl_error = GL_NO_ERROR; // Error returned by glGetError
SceGxmShaderPatcher *gxm_shader_patcher; // sceGxmShaderPatcher shader patcher instance
GLboolean is_fbo_float = GL_FALSE; // Current framebuffer mode

#ifdef HAVE_PROFILING
uint32_t frame_profiler_cnt = 0;
uint32_t ffp_draw_profiler_cnt = 0;
uint32_t ffp_reload_profiler_cnt = 0;
uint32_t shaders_draw_profiler_cnt = 0;
uint32_t ffp_draw_cnt = 0;
uint32_t shaders_draw_cnt = 0;
static uint32_t gpu_stall_cnt = 0;
#endif

int DISPLAY_WIDTH; // Display width in pixels
int DISPLAY_HEIGHT; // Display height in pixels
int DISPLAY_STRIDE; // Display stride in pixels
int NEW_DISPLAY_WIDTH; // Requested new display width in pixels
int NEW_DISPLAY_HEIGHT; // Requested new display height in pixels
float DISPLAY_WIDTH_FLOAT; // Display width in pixels (float)
float DISPLAY_HEIGHT_FLOAT; // Display height in pixels (float)

GLboolean system_app_mode = GL_FALSE; // Flag for system app mode usage
static GLboolean gxm_initialized = GL_FALSE; // Current sceGxm state
GLboolean is_rendering_display = GL_TRUE; // Flag for when drawing without fbo is being performed

float *legacy_pool = NULL; // Mempool for GL1 immediate draw pipeline
float *legacy_pool_ptr = NULL; // Current address for vertices population for GL1 immediate draw pipeline
#ifndef SKIP_ERROR_HANDLING
float *legacy_pool_end = NULL; // Address of the end of the GL1 immediate draw pipeline vertex pool
#endif
uint32_t vgl_framecount = 0; // Current frame number since application started

void *frame_purge_list[FRAME_PURGE_FREQ][FRAME_PURGE_LIST_SIZE]; // Purge list for internal elements
void *frame_rt_purge_list[FRAME_PURGE_FREQ][FRAME_PURGE_RENDERTARGETS_LIST_SIZE]; // Purge list for rendertargets
int frame_purge_idx = 0; // Index for currently populatable purge list
int frame_elem_purge_idx = 0; // Index for currently populatable purge list element
int frame_rt_purge_idx = 0; // Index for currently populatable purge list rendertarget
static int frame_purge_clean_idx = 1;
SceUID gc_mutex[2];
static int gc_thread_priority = 0x10000100;
static int gc_thread_affinity = 0;
static uint8_t gxm_display_rt_size = 1; // Number of scenes per frame to use for the display rendertarget
#ifdef HAVE_PTHREAD
pthread_t gc_thread;
#else
SceUID gc_thread;
#endif

#ifdef HAVE_CPU_TRACER
int sceRazorCpuSync();
#endif

#ifdef HAVE_RAZOR
#define RAZOR_BUF_SIZE (1024 * 1024) // Size in bytes for a live metrics data buffer
#define UPDATE_RATIO 30 // Number of frames between two live metrics updates
#ifndef HAVE_DEVKIT
#define RAZOR_CAPTURE_MOD_PATH "ur0:data/librazorcapture_es4.suprx"
SceUID razor_modid;
#endif

typedef union {
	SceRazorGpuLiveEntryJob *job;
	SceRazorGpuLiveEntryParameterBuffer *pbuf;
	SceRazorGpuLiveEntryFrame *frame;
	uintptr_t ptr;
} SceRazorGpuResult;

uint8_t *razor_buf[DISPLAY_MAX_BUFFER_COUNT]; // Buffers used to store live metrics data
razor_results razor_metrics;

GLboolean has_razor_live = GL_FALSE; // Flag for live metrics support with sceRazor
#endif

static inline __attribute__((always_inline)) int setupRenderTarget(SceGxmRenderTarget **rt, int w, int h, int refs) {
	SceGxmRenderTargetParams renderTargetParams;
	vgl_memset(&renderTargetParams, 0, sizeof(SceGxmRenderTargetParams));
	renderTargetParams.width = w ? w : 1;
	renderTargetParams.height = h ? h : 1;
	renderTargetParams.scenesPerFrame = refs;
	renderTargetParams.multisampleMode = msaa_mode;
	renderTargetParams.driverMemBlock = -1;
	return sceGxmCreateRenderTarget(&renderTargetParams, rt);
}

#ifdef HAVE_SHARED_RENDERTARGETS
#define MAX_RENDER_TARGETS_NUM 47 // Maximum amount of dedicated render targets usable for fbos
#define MAX_SHARED_RT_SIZE 256 // Maximum  width value in pixels for shared rendertargets usage
render_target rt_list[MAX_RENDER_TARGETS_NUM];

render_target *getFreeRenderTarget(int w, int h) {
	int i;
	for (i = 0; i < MAX_RENDER_TARGETS_NUM; i++) {
		if (rt_list[i].rt != NULL) {
			if (w == rt_list[i].w && h == rt_list[i].h && rt_list[i].ref_count < rt_list[i].max_refs) {
				rt_list[i].ref_count++;
				return &rt_list[i];
			}
		} else {
			rt_list[i].max_refs = w > MAX_SHARED_RT_SIZE ? 1 : MAX_SCENES_PER_FRAME;
			int r = setupRenderTarget(&rt_list[i].rt, w, h, rt_list[i].max_refs);
#ifdef LOG_ERRORS
			if (r)
				vgl_log("%s:%d Failed to create a shared rendertarget of size %dx%d (%s).\n", __FILE__, __LINE__, w, h, get_gxm_error_literal(r));
#endif
			rt_list[i].w = w;
			rt_list[i].h = h;
			rt_list[i].ref_count = 1;
			return &rt_list[i];
		}
	}
#ifdef RECYCLE_RENDERTARGETS
	vgl_log("%s:%d Out of rendertargets handles: Recycling an old rendertarget.\n", __FILE__, __LINE__);
	uint32_t oldest_framecount = 0xFFFFFFFF;
	render_target *r = NULL;
	for (i = 0; i < MAX_RENDER_TARGETS_NUM; i++) {
		if (rt_list[i].last_frame < oldest_framecount) {
			oldest_framecount = rt_list[i].last_frame;
			r = &rt_list[i];
		}
	}
	sceGxmFinish(gxm_context);
	sceGxmDestroyRenderTarget(r->rt);
	r->max_refs = w > MAX_SHARED_RT_SIZE ? 1 : MAX_SCENES_PER_FRAME;
	int res = setupRenderTarget(&r->rt, w, h, r->max_refs);
#ifdef LOG_ERRORS
	if (res)
		vgl_log("%s:%d Failed to create a shared rendertarget of size %dx%d (%s).\n", __FILE__, __LINE__, w, h, get_gxm_error_literal(res));
#endif
	r->w = w;
	r->h = h;
	r->ref_count = 1;
	return r;
#else
#ifdef LOG_ERRORS
	vgl_log("%s:%d Failed to create a shared rendertarget of size %dx%d (Out of rendertargets handles).\n", __FILE__, __LINE__, w, h);
#endif
	return NULL;
#endif
}

void __markRtAsDirty(render_target *rt) {
	rt->ref_count--;
	if (!rt->ref_count) {
		_markRtAsDirty(rt->rt);
		rt->rt = NULL;
	}
}
#endif

// sceDisplay callback data
struct display_queue_callback_data {
	void *addr;
};

// sceGxmShaderPatcher custom allocator
static void *shader_patcher_host_alloc_cb(void *user_data, unsigned int size) {
	return vglMalloc(size);
}

// sceGxmShaderPatcher custom deallocator
static void shader_patcher_host_free_cb(void *user_data, void *mem) {
	vgl_free(mem);
}

// sceDisplay callback
static void display_queue_callback(const void *callbackData) {
	// Populating sceDisplay framebuffer parameters
	SceDisplayFrameBuf display_fb;
	const struct display_queue_callback_data *cb_data = callbackData;
	vgl_memset(&display_fb, 0, sizeof(SceDisplayFrameBuf));
	display_fb.size = sizeof(SceDisplayFrameBuf);
	display_fb.base = cb_data->addr;
	display_fb.pitch = DISPLAY_STRIDE;
	display_fb.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
	display_fb.width = DISPLAY_WIDTH;
	display_fb.height = DISPLAY_HEIGHT;

#ifdef HAVE_DEBUG_INTERFACE
	// Drawing lightweighted debugger info
	vgl_debugger_draw(cb_data->addr);
#endif

	if (vgl_display_cb)
		vgl_display_cb(cb_data->addr);

	// Setting sceDisplay framebuffer
	sceDisplaySetFrameBuf(&display_fb, SCE_DISPLAY_SETBUF_NEXTFRAME);

	// Performing VSync if enabled
	if (vsync_interval)
		sceDisplayWaitVblankStartMulti(vsync_interval);
}

// Garbage collector
#if defined(HAVE_PTHREAD) && !defined(HAVE_SINGLE_THREADED_GC)
void garbage_collector(void *arg) {
#else
int garbage_collector(unsigned int args, void *arg) {
#endif
#ifndef HAVE_SINGLE_THREADED_GC
	for (;;) {
		// Waiting for garbage collection request
		sceKernelWaitSema(gc_mutex[0], 1, NULL);
#endif
		// Purging all elements marked for deletion
		for (int i = 0; i < FRAME_PURGE_LIST_SIZE; i++) {
			if (frame_purge_list[frame_purge_clean_idx][i]) {
				vgl_free(frame_purge_list[frame_purge_clean_idx][i]);
				frame_purge_list[frame_purge_clean_idx][i] = NULL;
			} else
				break;
		}
		for (int i = 0; i < FRAME_PURGE_RENDERTARGETS_LIST_SIZE; i++) {
			if (frame_rt_purge_list[frame_purge_clean_idx][i]) {
				sceGxmDestroyRenderTarget(frame_rt_purge_list[frame_purge_clean_idx][i]);
				frame_rt_purge_list[frame_purge_clean_idx][i] = NULL;
			} else
				break;
		}
		frame_purge_clean_idx = (frame_purge_clean_idx + 1) % FRAME_PURGE_FREQ;
		frame_purge_idx = (frame_purge_idx + 1) % FRAME_PURGE_FREQ;
		frame_elem_purge_idx = 0;
		frame_rt_purge_idx = 0;
#ifndef HAVE_SINGLE_THREADED_GC
		sceKernelSignalSema(gc_mutex[1], 1);
	}
#ifndef HAVE_PTHREAD
	return sceKernelExitDeleteThread(0);
#endif
#endif
}

GLboolean startShaderCompiler(void) {
	shark_set_allocators(vglMalloc, vglFree);
#ifdef HAVE_VITA3K_SUPPORT
	is_shark_online = shark_init_simple(NULL) >= 0;
#else
	is_shark_online = shark_init(NULL) >= 0;
#endif

	// If standard path failed to init we try to init it with ScePiglet path
	if (!is_shark_online) {
#ifdef HAVE_VITA3K_SUPPORT
		is_shark_online = shark_init_simple("ur0:data/external/libshacccg.suprx") >= 0;
#else
		is_shark_online = shark_init("ur0:data/external/libshacccg.suprx") >= 0;
#endif
#ifdef LOG_ERRORS
		if (!is_shark_online)
			vgl_log("%s:%d Fatal error: SceShaccCg not found.\n", __FILE__, __LINE__);
#endif
	}

	return is_shark_online;
}

void initGxm(void) {
	if (gxm_initialized)
		return;

#ifdef HAVE_RAZOR
	// Initializing sceRazor debugger
#ifdef HAVE_DEVKIT
	sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_HUD);
	sceSysmoduleLoadModule(SCE_SYSMODULE_RAZOR_CAPTURE);
#else
	razor_modid = sceKernelLoadStartModule(RAZOR_CAPTURE_MOD_PATH, 0, NULL, 0, NULL, NULL);
#endif
#ifdef HAVE_DEVKIT
	for (int i = 0; i < DISPLAY_MAX_BUFFER_COUNT; i++) {
		razor_buf[i] = vglMemalign(8, RAZOR_BUF_SIZE);
	}
#endif

	sceRazorGpuCaptureEnableSalvage("ux0:data/vitagl_gpucrash.sgx");
#endif

	// Initializing runtime shader compiler
	if (startShaderCompiler()) {
#if defined(HAVE_SHARK_LOG) || defined(LOG_ERRORS)
		shark_install_log_cb(shark_log_cb);
		shark_set_warnings_level(SHARK_WARN_HIGH);
#endif
	}

#ifndef HAVE_SINGLE_THREADED_GC
	// Initializing garbage collector
	gc_mutex[0] = sceKernelCreateSema("GC Sema Push", 0, 0, FRAME_PURGE_FREQ, NULL);
	gc_mutex[1] = sceKernelCreateSema("GC Sema Pull", 0, FRAME_PURGE_FREQ, FRAME_PURGE_FREQ, NULL);
#ifdef HAVE_PTRHEAD
	pthread_create(&gc_thread, NULL, garbage_collector, NULL);
	pthread_setaffinity_np(gc_thread, 4, &gc_thread_affinity);
#else
	gc_thread = sceKernelCreateThread("Garbage Collector", &garbage_collector, gc_thread_priority, 0x10000, 0, gc_thread_affinity, NULL);
	sceKernelStartThread(gc_thread, 0, NULL);
#endif
#endif

#ifndef HAVE_VITA3K_SUPPORT // Vita3K lacks sceGxmVshInitialize support, so we can't use it for sysapps
	// Checking if the running application is a system one
	SceAppMgrBudgetInfo info;
	info.size = sizeof(SceAppMgrBudgetInfo);
	if (!sceAppMgrGetBudgetInfo(&info)) {
		system_app_mode = GL_TRUE;
		gxm_display_buffer_count = 2; // Forcing double buffering in system app mode
		if (msaa_mode == SCE_GXM_MULTISAMPLE_NONE) // FIXME: For some reasons, disabling MSAA makes the shader patcher not able to compile fragment programs in sysapp mode...
			msaa_mode = SCE_GXM_MULTISAMPLE_2X;
	}
#endif

	// Initializing sceGxm init parameters
	SceGxmInitializeParams gxm_init_params;
	vgl_memset(&gxm_init_params, 0, sizeof(SceGxmInitializeParams));
#ifdef HAVE_VITA3K_SUPPORT // Vita3K lacks sceGxmVshInitialize support, so we use sceGxmInitialize instead and disable a couple of features (HW ETC1 support and sysapp mode support)
	gxm_init_params.flags = SCE_GXM_INITIALIZE_FLAG_DEFAULT;
#else
	gxm_init_params.flags = SCE_GXM_INITIALIZE_FLAG_EXTENDED_FORMAT;
	if (system_app_mode) {
		gxm_init_params.flags |= (SCE_GXM_INITIALIZE_FLAG_PB_LPDDR | SCE_GXM_INITIALIZE_FLAG_SHARED_SYNC | SCE_GXM_INITIALIZE_FLAG_SHAREDPB_CREATE | SCE_GXM_INITIALIZE_FLAG_SHAREDPB_OPEN);
	}
#endif
	gxm_init_params.displayQueueMaxPendingCount = gxm_display_buffer_count - 1;
	gxm_init_params.displayQueueCallback = display_queue_callback;
	gxm_init_params.displayQueueCallbackDataSize = sizeof(struct display_queue_callback_data);
	gxm_init_params.parameterBufferSize = gxm_param_buf_size;

	// Initializing sceGxm
#ifdef HAVE_VITA3K_SUPPORT // Vita3K lacks sceGxmVshInitialize support, so we use sceGxmInitialize instead and disable a couple of features (HW ETC1 support and sysapp mode support)
	sceGxmInitialize(&gxm_init_params);
#else
	sceGxmVshInitialize(&gxm_init_params);
#endif
	gxm_initialized = GL_TRUE;

#ifdef HAVE_DEVKIT
	sceRazorGpuLiveSetMetricsGroup(SCE_RAZOR_GPU_LIVE_METRICS_GROUP_PBUFFER_USAGE);
	has_razor_live = !sceRazorGpuLiveStart();
#endif
}

void initGxmContext(void) {
	// Allocating VDM ring buffer
	vdm_ring_buffer_addr = gpu_alloc_mapped_aligned(4096, gxm_vdm_buf_size, VGL_MEM_VRAM);

	// Allocating vertex ring buffer
	vertex_ring_buffer_addr = gpu_alloc_mapped_aligned(4096, gxm_vertex_buf_size, VGL_MEM_VRAM);

	// Allocating fragment ring buffer
	fragment_ring_buffer_addr = gpu_alloc_mapped_aligned(4096, gxm_fragment_buf_size, VGL_MEM_VRAM);

	// Allocating fragment USSE ring buffer
	unsigned int fragment_usse_offset;
	fragment_usse_ring_buffer_addr = gpu_fragment_usse_alloc_mapped(gxm_usse_buf_size, &fragment_usse_offset);

	// Setting sceGxm context parameters
	SceGxmContextParams gxm_context_params;
	vgl_memset(&gxm_context_params, 0, sizeof(SceGxmContextParams));
	gxm_context_params.hostMem = vglMalloc(SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE);
	gxm_context_params.hostMemSize = SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE;
	gxm_context_params.vdmRingBufferMem = vdm_ring_buffer_addr;
	gxm_context_params.vdmRingBufferMemSize = gxm_vdm_buf_size;
	gxm_context_params.vertexRingBufferMem = vertex_ring_buffer_addr;
	gxm_context_params.vertexRingBufferMemSize = gxm_vertex_buf_size;
	gxm_context_params.fragmentRingBufferMem = fragment_ring_buffer_addr;
	gxm_context_params.fragmentRingBufferMemSize = gxm_fragment_buf_size;
	gxm_context_params.fragmentUsseRingBufferMem = fragment_usse_ring_buffer_addr;
	gxm_context_params.fragmentUsseRingBufferMemSize = gxm_usse_buf_size;
	gxm_context_params.fragmentUsseRingBufferOffset = fragment_usse_offset;

	// Initializing sceGxm context
	sceGxmCreateContext(&gxm_context_params, &gxm_context);
#ifdef DISABLE_W_CLAMPING
	sceGxmSetWClampEnable(gxm_context, SCE_GXM_WCLAMP_MODE_DISABLED);
#endif

	// Initializing circular pool for uniform buffers
	vglSetupUniformCircularPool();
}

void createDisplayRenderTarget(void) {
	// Creating render target for the display
	setupRenderTarget(&gxm_render_target, DISPLAY_WIDTH, DISPLAY_HEIGHT, gxm_display_rt_size);
}

void initDisplayColorSurfaces(GLboolean is_swap) {
	// Getting access to the shared framebuffer on system app mode
	while (system_app_mode) {
		shared_fb = sceSharedFbOpen(1);
		vgl_memset(&shared_fb_info, 0, sizeof(SceSharedFbInfo));
		sceSharedFbGetInfo(shared_fb, &shared_fb_info);
		if (shared_fb_info.index == 1)
			sceSharedFbClose(shared_fb);
		else {
			sceGxmMapMemory(shared_fb_info.fb_base, shared_fb_info.fb_size, SCE_GXM_MEMORY_ATTRIB_READ | SCE_GXM_MEMORY_ATTRIB_WRITE);
			gxm_color_surfaces_addr[0] = shared_fb_info.fb_base;
			gxm_color_surfaces_addr[1] = shared_fb_info.fb_base2;
			break;
		}
	}

	for (int i = 0; i < gxm_display_buffer_count; i++) {
		// Allocating color surface memblock
		if (!system_app_mode) {
			gxm_color_surfaces_addr[i] = gpu_alloc_mapped_aligned(4096, VGL_ALIGN(4 * DISPLAY_STRIDE * DISPLAY_HEIGHT, 1 * 1024 * 1024), VGL_MEM_VRAM);
			vgl_memset(gxm_color_surfaces_addr[i], 0, 4 * DISPLAY_STRIDE * DISPLAY_HEIGHT);
		}

		// Initializing allocated color surface
		sceGxmColorSurfaceInit(&gxm_color_surfaces[i],
			SCE_GXM_COLOR_FORMAT_A8B8G8R8,
			SCE_GXM_COLOR_SURFACE_LINEAR,
			msaa_mode == SCE_GXM_MULTISAMPLE_NONE ? SCE_GXM_COLOR_SURFACE_SCALE_NONE : SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE,
			SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
			DISPLAY_WIDTH,
			DISPLAY_HEIGHT,
			DISPLAY_STRIDE,
			gxm_color_surfaces_addr[i]);

		// Creating a display sync object for the allocated color surface
		if (!is_swap)
			sceGxmSyncObjectCreate(&gxm_sync_objects[i]);
	}
}

void initDepthStencilBuffer(uint32_t w, uint32_t h, SceGxmDepthStencilSurface *surface, GLboolean has_stencil) {
	// Calculating sizes for depth and stencil surfaces
	unsigned int depth_stencil_width = VGL_ALIGN(w, SCE_GXM_TILE_SIZEX);
#ifndef DEPTH_STENCIL_HACK
	unsigned int depth_stencil_height = VGL_ALIGN(h, SCE_GXM_TILE_SIZEY);
	unsigned int depth_stencil_samples = depth_stencil_width * depth_stencil_height;
	if (msaa_mode == SCE_GXM_MULTISAMPLE_2X)
		depth_stencil_samples *= 2;
	else if (msaa_mode == SCE_GXM_MULTISAMPLE_4X)
		depth_stencil_samples *= 4;

	// Allocating depth surface
	void *depth_buffer = gpu_alloc_mapped_aligned(SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT, 4 * depth_stencil_samples, VGL_MEM_VRAM);

#ifdef STORE_DEPTH_STENCIL
	// Initializing mask update bit to 1
	vgl_memset(depth_buffer, 0x80, 4 * depth_stencil_samples);
#endif
#endif

	// Allocating stencil surface
#ifndef DEPTH_STENCIL_HACK
	void *stencil_buffer = NULL;
	if (has_stencil)
		stencil_buffer = gpu_alloc_mapped_aligned(SCE_GXM_DEPTHSTENCIL_SURFACE_ALIGNMENT, depth_stencil_samples, VGL_MEM_VRAM);
#endif

	// Initializing depth and stencil surfaces
	vglDepthStencilSurfaceInit(surface,
		has_stencil ? SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M_S8 : SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M,
		SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR,
		msaa_mode == SCE_GXM_MULTISAMPLE_4X ? depth_stencil_width * 2 : depth_stencil_width,
#ifdef DEPTH_STENCIL_HACK
		// Vita's GPU can run without actual depth/stencil memory as far as no partial rendering is hit
		NULL, NULL);
#else
		depth_buffer, stencil_buffer);
#endif

#ifdef STORE_DEPTH_STENCIL
	sceGxmDepthStencilSurfaceSetForceLoadMode(surface, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
	sceGxmDepthStencilSurfaceSetForceStoreMode(surface, SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);
#endif
}

void initDepthStencilSurfaces(void) {
	initDepthStencilBuffer(DISPLAY_WIDTH, DISPLAY_HEIGHT, &gxm_depth_stencil_surface, GL_TRUE);
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	if (display_depth_persist) {
		// vglFlushFrame support: every display scene force-STORES depth/stencil at its EndScene so a
		// mid-frame scene split can resume with the depth accumulated so far. The matching force-LOAD is
		// one-shot (armed by vglFlushFrame, consumed at the restart's sceGxmBeginScene in sceneReset), so
		// ordinary frame-start scenes keep the cheap cleared-tile start. GXM latches both flags at
		// BeginScene, which is why store must be permanently armed here rather than at split time.
		// Mask-update bits must be initialized to 1 (0x80 byte pattern) or the masked DF32M store would
		// skip pixels — mirrors the STORE_DEPTH_STENCIL init in initDepthStencilBuffer.
		unsigned int depth_stencil_width = VGL_ALIGN(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX);
		unsigned int depth_stencil_height = VGL_ALIGN(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY);
		unsigned int depth_stencil_samples = depth_stencil_width * depth_stencil_height;
		if (msaa_mode == SCE_GXM_MULTISAMPLE_2X)
			depth_stencil_samples *= 2;
		else if (msaa_mode == SCE_GXM_MULTISAMPLE_4X)
			depth_stencil_samples *= 4;
		vgl_memset(gxm_depth_stencil_surface.depthData, 0x80, 4 * depth_stencil_samples);
		sceGxmDepthStencilSurfaceSetForceStoreMode(&gxm_depth_stencil_surface, SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);
	}
#endif
}

void startShaderPatcher(void) {
	// Allocating Shader Patcher buffer
	gxm_shader_patcher_buffer_addr = gpu_alloc_mapped_aligned(4, shader_patcher_buffer_size, VGL_MEM_VRAM);

	// Allocating Shader Patcher vertex USSE buffer
	unsigned int shader_patcher_vertex_usse_offset;
	gxm_shader_patcher_vertex_usse_addr = gpu_vertex_usse_alloc_mapped(shader_patcher_vertex_usse_size, &shader_patcher_vertex_usse_offset);

	// Allocating Shader Patcher fragment USSE buffer
	unsigned int shader_patcher_fragment_usse_offset;
	gxm_shader_patcher_fragment_usse_addr = gpu_fragment_usse_alloc_mapped(shader_patcher_fragment_usse_size, &shader_patcher_fragment_usse_offset);

	// Populating shader patcher parameters
	SceGxmShaderPatcherParams shader_patcher_params;
	vgl_memset(&shader_patcher_params, 0, sizeof(SceGxmShaderPatcherParams));
	shader_patcher_params.userData = NULL;
	shader_patcher_params.hostAllocCallback = shader_patcher_host_alloc_cb;
	shader_patcher_params.hostFreeCallback = shader_patcher_host_free_cb;
	shader_patcher_params.bufferAllocCallback = NULL;
	shader_patcher_params.bufferFreeCallback = NULL;
	shader_patcher_params.bufferMem = gxm_shader_patcher_buffer_addr;
	shader_patcher_params.bufferMemSize = shader_patcher_buffer_size;
	shader_patcher_params.vertexUsseAllocCallback = NULL;
	shader_patcher_params.vertexUsseFreeCallback = NULL;
	shader_patcher_params.vertexUsseMem = gxm_shader_patcher_vertex_usse_addr;
	shader_patcher_params.vertexUsseMemSize = shader_patcher_vertex_usse_size;
	shader_patcher_params.vertexUsseOffset = shader_patcher_vertex_usse_offset;
	shader_patcher_params.fragmentUsseAllocCallback = NULL;
	shader_patcher_params.fragmentUsseFreeCallback = NULL;
	shader_patcher_params.fragmentUsseMem = gxm_shader_patcher_fragment_usse_addr;
	shader_patcher_params.fragmentUsseMemSize = shader_patcher_fragment_usse_size;
	shader_patcher_params.fragmentUsseOffset = shader_patcher_fragment_usse_offset;

	// Creating shader patcher instance
	sceGxmShaderPatcherCreate(&shader_patcher_params, &gxm_shader_patcher);
}

void stopShaderPatcher(void) {
	// Destroying shader patcher instance
	sceGxmShaderPatcherDestroy(gxm_shader_patcher);

	// Freeing shader patcher buffers
	vgl_free(gxm_shader_patcher_buffer_addr);
	gpu_vertex_usse_free_mapped(gxm_shader_patcher_vertex_usse_addr);
	gpu_fragment_usse_free_mapped(gxm_shader_patcher_fragment_usse_addr);
}

static inline __attribute__((always_inline)) void sceneEnd(void) {
	// [EFB-ASYNC v3] WAR/consumer fence: a pending capture transfer READS the display color surface and
	// WRITES a capture texture; the fragment job THIS EndScene submits may write that same surface (the
	// post-capture restart scene, incl. the copy-clear) and/or sample that texture. The transfer queue is
	// NOT ordered against the render queue (device-proven: unfenced v1 and display-sync-object-fenced v2
	// both flickered stale on alternating frames then wedged the GPU), so CPU-wait the transfer-done
	// notification BEFORE submitting any fragment job. This is the FIRST point the hazard can exist —
	// fragment jobs only ever enter the queue through this sceGxmEndScene (all four sceneEnd call sites).
	// By here the tiny capture DMA was enqueued ~a scene's worth of CPU work ago, so the wait is
	// ~always already signaled (instrumented: vgl_efbwait_tn_*).
	if (capture_tn_pending) {
		capture_tn_pending = GL_FALSE;
		uint64_t tn_t0 = sceKernelGetProcessTimeWide();
		sceGxmNotificationWait(&capture_tn);
		uint32_t tn_dt = (uint32_t)(sceKernelGetProcessTimeWide() - tn_t0);
		vgl_efbwait_tn_us += tn_dt;
		vgl_efbwait_tn_cnt++;
		if (tn_dt > vgl_efbwait_tn_max)
			vgl_efbwait_tn_max = tn_dt;
	}
	// Ends current gxm scene
	query_fence.value++;
	const uint32_t es_t0 = sceKernelGetProcessTimeLow();
	sceGxmEndScene(gxm_context, NULL, &query_fence);
	vgl_ps_end_scene_us += sceKernelGetProcessTimeLow() - es_t0;
	vgl_ps_end_scenes++;
	if (system_app_mode && vsync_interval)
		sceDisplayWaitVblankStartMulti(vsync_interval);
}

void sceneReset(void) {
	if (in_use_framebuffer != active_write_fb || needs_scene_reset || dirty_framebuffer || dirty_query) {
		dirty_framebuffer = GL_FALSE;
		dirty_query = GL_FALSE;
		needs_scene_reset = GL_FALSE;
		in_use_framebuffer = active_write_fb;
		is_fbo_float = in_use_framebuffer ? in_use_framebuffer->is_float : GL_FALSE;

#ifdef DRAW_STATE_CACHE
		// sceGxmBeginScene below resets all GXM state, invalidating our cache
		extern void vgl_draw_state_cache_reset(void);
		vgl_draw_state_cache_reset();
#endif
		// ...and the default uniform buffer bindings with it: a versioned-upload reuse
		// (vglSetUniformVersioning) must bind its copy again even when vgl_def_*_buf
		// already names it. Every mid-frame restart (vglFlushFrame, vglSceneSubmit,
		// the captures, a framebuffer change) comes through here before its next draw.
		vgl_unif_bind_stale[0] = GL_TRUE;
		vgl_unif_bind_stale[1] = GL_TRUE;
		vgl_scene_epoch++;

		// Ending drawing scene
		if (needs_end_scene) {
			sceneEnd();
		} else {
			if (legacy_pool_size) {
				legacy_pool = (float *)gpu_alloc_mapped_temp(legacy_pool_size);
				legacy_pool_ptr = legacy_pool;
#ifndef SKIP_ERROR_HANDLING
				legacy_pool_end = (float *)((uint8_t *)legacy_pool + legacy_pool_size);
#endif
			}
			needs_end_scene = GL_TRUE;
		}

		// Starting drawing scene
		is_rendering_display = !active_write_fb;
		if (is_rendering_display) { // Default framebuffer is used
			if (system_app_mode) {
				sceSharedFbBegin(shared_fb, &shared_fb_info);
				shared_fb_info.vsync = vsync_interval;
				gxm_back_buffer_index = (shared_fb_info.index + 1) % 2;
			}
			const uint32_t bs_t0 = sceKernelGetProcessTimeLow();
#ifdef LOG_ERRORS
			int r = sceGxmBeginScene(gxm_context, 0, gxm_render_target,
				NULL, NULL,
				gxm_sync_objects[gxm_back_buffer_index],
				&gxm_color_surfaces[gxm_back_buffer_index],
				&gxm_depth_stencil_surface);
			if (r)
				vgl_log("%s:%d Scene reset failed due to sceGxmBeginScene erroring (%s) on display.\n", __FILE__, __LINE__, get_gxm_error_literal(r));
#else
			sceGxmBeginScene(gxm_context, 0, gxm_render_target,
				NULL, NULL,
				gxm_sync_objects[gxm_back_buffer_index],
				&gxm_color_surfaces[gxm_back_buffer_index],
				&gxm_depth_stencil_surface);
#endif
			vgl_ps_note_begin(bs_t0);
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
			// vglFlushFrame one-shot: the BeginScene above sampled the armed force-load (this restart
			// scene resumes the depth stored by the split's EndScene); disarm so later display scenes
			// go back to the default cleared-tile start.
			if (display_depth_load_pending) {
				display_depth_load_pending = GL_FALSE;
				sceGxmDepthStencilSurfaceSetForceLoadMode(&gxm_depth_stencil_surface, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
			}
#endif
		} else {
			// If a depthstencil surface is not bound to the in use framebuffer, we get one for it to ensure scissor testing compatibility
			if (!active_write_fb->depthbuffer_ptr) {
				initDepthStencilBuffer(active_write_fb->width, active_write_fb->height, &active_write_fb->depthbuffer, GL_FALSE);
				active_write_fb->depthbuffer_ptr = &active_write_fb->depthbuffer;
				active_write_fb->is_depth_hidden = GL_TRUE;
			}

			// If a rendertarget is not bound to the in use framebuffer, we get one for it
			if (!active_write_fb->target) {
#ifdef HAVE_SHARED_RENDERTARGETS
				active_write_fb->target = (SceGxmRenderTarget *)getFreeRenderTarget(active_write_fb->width, active_write_fb->height);
#else
				// [CAPFENCE] scenesPerFrame = MAX_SCENES_PER_FRAME, NOT 1. An FBO hosting mid-frame capture
				// splits (vglCaptureFboRegion) runs SEVERAL scenes per frame on this render target: the main
				// scene (ended at each split) + the restart scene (copy-clear/stencil/draws, ended at the
				// FBO->display transition) — and with the caller pipelined one frame ahead, the previous
				// frame's scenes may still be in flight when this frame's end-scene submits. Exceeding the
				// DECLARED scenesPerFrame makes sceGxmEndScene/BeginScene block inside the driver on scene-
				// resource recycling — the only un-instrumented seconds-capable wait in the capture path
				// (device: bit27 transition captures measured 1.845-1.875s inside vglCaptureFboRegion while
				// the instrumented FN/TN fences read 7.1ms/3us — the fence was NOT the stall; and the block
				// self-feeds, since the recycling cadence is itself throttled by the stalled frame). This is
				// the FBO twin of the display-RT rule already documented at vglFlushFrame: "the display
				// render target must have scene headroom for the extra Begin/End pairs
				// (vglSetupDisplayRenderTarget)". The HAVE_SHARED_RENDERTARGETS build (above) already uses
				// MAX_SCENES_PER_FRAME for FBO-class RTs; this aligns the dedicated-RT path. Cost:
				// driver-internal per-scene bookkeeping only (driverMemBlock = -1 auto-alloc).
				int r = setupRenderTarget(&active_write_fb->target, active_write_fb->width, active_write_fb->height, MAX_SCENES_PER_FRAME);
#ifdef LOG_ERRORS
				if (r)
					vgl_log("%s:%d Failed to create a rendertarget of size %dx%d for framebuffer 0x%08X (%s).\n", __FILE__, __LINE__, active_write_fb->width, active_write_fb->height, active_write_fb, get_gxm_error_literal(r));
#endif
#endif
			}
#ifdef RECYCLE_RENDERTARGETS
			else {
				render_target *fbo_rt = (render_target *)active_write_fb->target;
				if (active_write_fb->width != fbo_rt->w || active_write_fb->height != fbo_rt->h) {
					vgl_log("%s:%d Attempting to use a recycled rendertarget. Re-allocating it.\n", __FILE__, __LINE__);
					active_write_fb->target = (SceGxmRenderTarget *)getFreeRenderTarget(active_write_fb->width, active_write_fb->height);
				}
			}
#endif
#ifdef HAVE_TEX_CACHE
			// FIXME: This may be useful even without texture cache enabled maybe?
			if (sceGxmColorSurfaceGetData(&active_write_fb->colorbuffer) != active_write_fb->tex->data)
				sceGxmColorSurfaceSetData(&active_write_fb->colorbuffer, active_write_fb->tex->data);
#endif
			const uint32_t bs_t0 = sceKernelGetProcessTimeLow();
#ifdef HAVE_SHARED_RENDERTARGETS
			render_target *fbo_rt = (render_target *)active_write_fb->target;
#ifdef RECYCLE_RENDERTARGETS
			fbo_rt->last_frame = vgl_framecount;
#endif
			int r = sceGxmBeginScene(gxm_context, 0, fbo_rt->rt,
#else
			int r = sceGxmBeginScene(gxm_context, 0, active_write_fb->target,
#endif
					NULL, NULL, NULL,
					&active_write_fb->colorbuffer,
					active_write_fb->depthbuffer_ptr);
			vgl_ps_note_begin(bs_t0);
#ifdef LOG_ERRORS
			if (r)
				vgl_log("%s:%d Scene reset failed due to sceGxmBeginScene erroring (%s) on framebuffer 0x%08X.\n", __FILE__, __LINE__, get_gxm_error_literal(r), active_write_fb);
#endif
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
			// [FBORT] vglCaptureFboRegion one-shot: the BeginScene above sampled the armed depth force-LOAD
			// (this restart resumes the depth the capture split stored); disarm it so a later restart of THIS
			// same FBO doesn't keep force-loading, and the next frame-start clear starts fresh. The force-STORE
			// stays armed on the surface for the rest of the frame (subsequent splits keep continuity).
			if (fbo_depth_load_pending && fbo_depth_load_fb == active_write_fb && active_write_fb->depthbuffer_ptr) {
				fbo_depth_load_pending = GL_FALSE;
				fbo_depth_load_fb = NULL;
				sceGxmDepthStencilSurfaceSetForceLoadMode(active_write_fb->depthbuffer_ptr, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_DISABLED);
			}
#endif
		}

		// Setting back current viewport if enabled cause sceGxm will reset it at sceGxmEndScene call
		if (old_framebuffer != in_use_framebuffer) {
			old_framebuffer = in_use_framebuffer;
			glViewport(gl_viewport.x, gl_viewport.y, gl_viewport.w, gl_viewport.h);
			skip_scene_reset = GL_TRUE;
			glScissor(region.gl_x, region.gl_y, region.gl_w, region.gl_h);
			skip_scene_reset = GL_FALSE;
#ifndef HAVE_UNFLIPPED_FBOS
			change_cull_mode();
#endif
		} else
			setViewport(gxm_context, x_port, x_scale, y_port, y_scale, z_port, z_scale);

#ifndef DISABLE_TILE_CLIPPER
		if (scissor_test_state)
			sceGxmSetRegionClip(gxm_context, SCE_GXM_REGION_CLIP_OUTSIDE, region.x, region.y, region.x + region.w - 1, region.y + region.h - 1);
#endif
	}
}

/*
 * ------------------------------
 * - IMPLEMENTATION STARTS HERE -
 * ------------------------------
 */

void vglSetupGarbageCollector(int priority, int affinity) {
	gc_thread_priority = priority;
	gc_thread_affinity = affinity;
}

void vglSetParamBufferSize(uint32_t size) {
	gxm_param_buf_size = size;
}

void vglSetVDMBufferSize(uint32_t size) {
	gxm_vdm_buf_size = size;
}

void vglSetVertexBufferSize(uint32_t size) {
	gxm_vertex_buf_size = size;
}

void vglSetFragmentBufferSize(uint32_t size) {
	gxm_fragment_buf_size = size;
}

void vglSetUSSEBufferSize(uint32_t size) {
	gxm_usse_buf_size = size;
}

void vglUseTripleBuffering(GLboolean usage) {
	gxm_display_buffer_count = usage ? 3 : 2;
}

void vglSetDisplayBufferCount(int count) {
#ifndef SKIP_ERROR_HANDLING
	if (count < 0) {
		SET_GL_ERROR_WITH_VALUE(GL_INVALID_VALUE, count)
	}
#endif
	gxm_display_buffer_count = count;
}

// [FSKIP4] one-shot "present without flip" latch, consumed by the display-queue block inside
// vglSwapBuffers below. Set ONLY by vglSwapBuffersSkipFlip; entry-latched (read+cleared at the top of
// vglSwapBuffers) so it can never leak across a present that doesn't reach the display-queue block
// (FBO-bound present / system-app mode).
static GLboolean vgl_skip_next_flip = GL_FALSE;

// [FSKIP4/FSKIP5] Present WITHOUT flipping to display: identical to vglSwapBuffers(GL_FALSE) — same
// vgl_framecount++/circular-pool rotation, same dirty_frag_unifs/dirty_vert_unifs marking, same
// draw-state-cache reset, same sceneEnd() (incl. the EFB-ASYNC capture TN wait + sceGxmEndScene), same
// needs_scene_reset, same resolution-change handling and garbage-collector kick — EXCEPT the
// display-queue entry block: no sceGxmDisplayQueueAddEntry (no flip, no sync-object display pairing,
// no vsync wait in the queue callback for this frame) AND [FSKIP5] NO BUFFER ROTATION AT ALL:
// gxm_front_buffer_index stays put (it must keep naming the buffer ACTUALLY on display so the next
// real flip pairs its AddEntry against the correct "old buffer" sync object) and gxm_back_buffer_index
// ALSO stays put — the skip-flip frame's content is never displayed, so the NEXT tick simply
// RE-RENDERS THE SAME BUFFER, overwriting undisplayed pixels.
//
// WHY NO ROTATION ([FSKIP5] — device post-mortem of the FSKIP4 rotate-without-flip version): rotating
// per present consumes buffers at TICK rate (2 per 33.3ms in the alternating flip/skip hold cadence)
// while AddEntry frees them at FLIP rate (1 per 33.3ms at vsync_interval=2) — a structural starvation
// no ordering enumeration can fix: renders reached buffers whose previous display term had not ended,
// their fragment jobs stalled on the display sync object, and the pipeline self-fed into collapse
// (device: 204ms FN fence waits, the board-select crossfade at 1.65fps/3.3 ticks, 2.3MB vrt backlog).
// With no rotation, buffers are consumed exactly at flip rate = freed rate: the STOCK triple-buffer
// reuse distance is restored, identical to non-skip operation.
//
// SAFETY of the same-buffer re-render:
//   * Consecutive BeginScene on the same index/color surface is the ESTABLISHED scene-split pattern:
//     vglFlushFrame and vglSceneSubmit (below) both end the display scene and arm "the next draw
//     re-begins on the same gxm_back_buffer_index" (their own comments) — exercised several times per
//     frame by the EFB capture machinery on real hardware. Same-context fragment jobs execute in
//     submission order, so the overwrite cannot race the previous scene's job.
//   * A pending EFB capture transfer OUT of this buffer cannot be raced either: sceneEnd()'s
//     capture_tn WAR fence blocks the next fragment job until the transfer completed.
//   * The buffer's sync object holds no pending display op (this frame was never queued), so the
//     re-render's BeginScene(fragmentSyncObject) neither stalls nor races scan-out.
//   * gxm_color_surfaces_addr[gxm_back_buffer_index] (the EFB capture source) keeps naming the buffer
//     being drawn — trivially stable across the non-rotating present.
//   * Depth: gxm_depth_stencil_surface is shared, not per-buffer; a skip-flip present leaves it
//     exactly as a normal swap does (no force-load armed; the caller's frame-start clear rebuilds it).
//
// HOLD-MODE BUFFER LIFECYCLE (3 buffers; front=A displayed, back=B): flip B [front=B back=C] ->
// skip C [no rotation] -> flip C with fresh content [front=C back=A] -> skip A -> flip A [front=A
// back=B] -> skip B -> flip B ... — every flip target was freed by an AddEntry a FULL flip period
// (33.3ms) earlier, and every skip target is the just-flipped-from back buffer, off-display by
// construction. Zero structural waits at any hold length.
void vglSwapBuffersSkipFlip(void) {
	vgl_skip_next_flip = GL_TRUE;
	vglSwapBuffers(GL_FALSE);
}

void vglSwapBuffers(GLboolean has_commondialog) {
	// [FSKIP4] entry-latch the skip-flip request (see vglSwapBuffersSkipFlip above)
	GLboolean skip_flip = vgl_skip_next_flip;
	vgl_skip_next_flip = GL_FALSE;
#ifdef HAVE_PROFILING
	// Show profiling results once every 30 frames to not clog CPU
	uint32_t tick = sceKernelGetProcessTimeLow();
	static uint32_t frame_start_profiler_cnt = 0;
	frame_profiler_cnt += tick - frame_start_profiler_cnt;
	if ((vgl_framecount % 30) == 0) {
		vgl_log("-----------------------------------------\n");
		vgl_log("Last 30 frames took %ums to be processed.\n", frame_profiler_cnt / 1000);
		vgl_log("%ums spent processing %u fixed-function pipeline non-immediate draw calls.\n", ffp_draw_profiler_cnt / 1000, ffp_draw_cnt);
		vgl_log("%ums spent setting up fixed-function pipeline states.\n", ffp_reload_profiler_cnt / 1000);
		vgl_log("%ums spent processing %u shaders pipeline draw calls.\n", shaders_draw_profiler_cnt / 1000, shaders_draw_cnt);
		vgl_log("%ums spent waiting for GPU to process frames.\n", gpu_stall_cnt / 1000);
		vgl_log("-----------------------------------------\n");
		frame_profiler_cnt = 0;
		ffp_draw_profiler_cnt = 0;
		ffp_reload_profiler_cnt = 0;
		shaders_draw_profiler_cnt = 0;
		shaders_draw_cnt = 0;
		ffp_draw_cnt = 0;
		gpu_stall_cnt = 0;
	}
	frame_start_profiler_cnt = tick;
#endif

	vgl_framecount++;
#if !defined(DISABLE_CIRCULAR_POOL) && !defined(CIRCULAR_POOL_SPEEDHACK)
#ifdef HAVE_DEBUG_INTERFACE
	vgl_circular_pool_frame_peak = (uint32_t)circular_data_pool_ptr[vgl_circular_idx] - (uint32_t)circular_data_pool[vgl_circular_idx];
	if (vgl_circular_pool_frame_peak > vgl_circular_pool_global_peak) {
		vgl_circular_pool_global_peak = vgl_circular_pool_frame_peak;
	}
#endif
	vgl_circular_idx = vgl_framecount % gxm_display_buffer_count;
	circular_data_pool_ptr[vgl_circular_idx] = circular_data_pool[vgl_circular_idx];
#endif

	// Marking uniform values as dirty at each frame end just to be safe
	dirty_frag_unifs = GL_TRUE;
	dirty_vert_unifs = GL_TRUE;
	needs_end_scene = GL_FALSE;

#ifdef DRAW_STATE_CACHE
	// Invalidate the per-draw GXM state cache so the first draw of the
	// next frame re-issues all set commands.
	extern void vgl_draw_state_cache_reset(void);
	vgl_draw_state_cache_reset();
#endif
#ifdef UNIFORM_VALUE_CACHE
	// Invalidate per-program last buffer pointers. The circular pool
	// recycles across frames; a buffer saved from last frame may now
	// contain stale or reallocated data.
	extern void vgl_uniform_cache_reset(void);
	vgl_uniform_cache_reset();
#endif

	if (!needs_scene_reset)
		sceneEnd();

	if (has_commondialog) {
		// Populating SceCommonDialog parameters
		SceCommonDialogUpdateParam updateParam;
		vgl_memset(&updateParam, 0, sizeof(updateParam));
		updateParam.renderTarget.colorFormat = SCE_GXM_COLOR_FORMAT_A8B8G8R8;
		updateParam.renderTarget.surfaceType = SCE_GXM_COLOR_SURFACE_LINEAR;
		updateParam.renderTarget.width = DISPLAY_WIDTH;
		updateParam.renderTarget.height = DISPLAY_HEIGHT;
		updateParam.renderTarget.strideInPixels = DISPLAY_STRIDE;
		updateParam.renderTarget.colorSurfaceData = gxm_color_surfaces_addr[gxm_back_buffer_index];
		updateParam.renderTarget.depthSurfaceData = gxm_depth_stencil_surface.depthData;
		updateParam.displaySyncObject = gxm_sync_objects[gxm_back_buffer_index];

		// Updating sceCommonDialog
		sceCommonDialogUpdate(&updateParam);
	}

	if (!in_use_framebuffer) {
		if (system_app_mode)
			sceSharedFbEnd(shared_fb);
		else {
#ifdef HAVE_RAZOR
			sceGxmPadHeartbeat(&gxm_color_surfaces[gxm_back_buffer_index], gxm_sync_objects[gxm_back_buffer_index]);
#ifdef HAVE_DEVKIT
			if (has_razor_live) {
				SceRazorGpuLiveResultInfo razor_res;
				sceRazorGpuLiveSetBuffer(razor_buf[gxm_back_buffer_index], RAZOR_BUF_SIZE, &razor_res);

				if (razor_res.result_data) {
					if (!razor_res.overflow_count) {
						vgl_memset(&razor_metrics, 0, sizeof(razor_results));
						SceUID pid = sceKernelGetProcessId();
						SceRazorGpuResult r;
						r.ptr = (uintptr_t)razor_res.result_data;

						// Analyzing the collected jobs
						for (uint32_t i = 0; i < razor_res.entry_count; i++) {
							switch (r.job->header.entry_type) {
							case SCE_RAZOR_LIVE_TRACE_METRIC_ENTRY_TYPE_JOB:
								if ((pid == r.job->process_id) && (r.job->type != SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FIRMWARE)) {
									if (razor_metrics.scene_count < r.job->scene_index + 1)
										razor_metrics.scene_count = r.job->scene_index + 1;
									switch (r.job->type) {
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX0:
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX1:
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX2:
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX3:
										razor_metrics.vertex_job_count++;
										razor_metrics.vertex_job_time += r.job->end_time - r.job->start_time;
										if (r.job->scene_index < RAZOR_MAX_SCENES_NUM) {
											razor_metrics.scenes[r.job->scene_index].vertex_duration += r.job->end_time - r.job->start_time;
										}
										break;
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT0:
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT1:
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT2:
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT3:
										razor_metrics.fragment_job_count++;
										razor_metrics.fragment_job_time += r.job->end_time - r.job->start_time;
										if (r.job->scene_index < RAZOR_MAX_SCENES_NUM) {
											razor_metrics.scenes[r.job->scene_index].fragment_duration += r.job->end_time - r.job->start_time;
										}
										break;
									}
									switch (r.job->type) {
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX1:
										razor_metrics.usse_vertex_processing_percent += r.job->job_values.vertex_values_type1.usse_vertex_processing_percent;
										break;
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX2:
										razor_metrics.vdm_primitives_input_num += r.job->job_values.vertex_values_type2.vdm_primitives_input_num;
										razor_metrics.mte_primitives_output_num += r.job->job_values.vertex_values_type2.mte_primitives_output_num;
										razor_metrics.vdm_vertices_input_num += r.job->job_values.vertex_values_type2.vdm_vertices_input_num;
										razor_metrics.mte_vertices_output_num += r.job->job_values.vertex_values_type2.mte_vertices_output_num;
										break;
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_VERTEX3:
										razor_metrics.tiling_accelerated_mem_writes += r.job->job_values.vertex_values_type3.tiling_accelerated_mem_writes;
										break;
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT1:
										razor_metrics.usse_fragment_processing_percent += r.job->job_values.fragment_values_type1.usse_fragment_processing_percent;
										razor_metrics.usse_dependent_texture_reads_percent += r.job->job_values.fragment_values_type1.usse_dependent_texture_reads_percent;
										razor_metrics.usse_non_dependent_texture_reads_percent += r.job->job_values.fragment_values_type1.usse_non_dependent_texture_reads_percent;
										break;
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT2:
										razor_metrics.rasterized_pixels_before_hsr_num += r.job->job_values.fragment_values_type2.rasterized_pixels_before_hsr_num;
										razor_metrics.rasterized_output_pixels_num += r.job->job_values.fragment_values_type2.rasterized_output_pixels_num;
										razor_metrics.rasterized_output_samples_num += r.job->job_values.fragment_values_type2.rasterized_output_samples_num;
										break;
									case SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FRAGMENT3:
										razor_metrics.isp_parameter_fetches_mem_reads += r.job->job_values.fragment_values_type3.isp_parameter_fetches_mem_reads;
										break;
									}
								} else if (r.job->type == SCE_RAZOR_LIVE_TRACE_METRIC_JOB_TYPE_FIRMWARE) {
									razor_metrics.firmware_job_count++;
									razor_metrics.firmware_job_time += r.job->end_time - r.job->start_time;
								}
								break;
							case SCE_RAZOR_LIVE_TRACE_METRIC_ENTRY_TYPE_PARAMETER_BUFFER:
								vgl_fast_memcpy(&razor_metrics.peak_usage_value, &r.pbuf->peak_usage_value, 6);
								break;
							case SCE_RAZOR_LIVE_TRACE_METRIC_ENTRY_TYPE_FRAME:
								vgl_fast_memcpy(&razor_metrics.frame_start_time, &r.frame->start_time, 20);
								break;
							default:
								break;
							}
							r.ptr += r.job->header.entry_size;
						}
					} else {
						vgl_log("%s:%d Razor Live Metrics overflow detected (%d entries). Consider increasing RAZOR_BUF_SIZE.\n", __FILE__, __LINE__, razor_res.overflow_count);
					}
				}
			}
#endif
#endif
			if (skip_flip) {
				// [FSKIP5] present-without-flip: the scene into the current back buffer was fully
				// submitted (sceneEnd above) but is NOT queued for display — no AddEntry, no vsync
				// wait for this frame, and NO index movement: the front buffer keeps displaying (and
				// keeps naming the on-display buffer for the next real flip's "old buffer" pairing),
				// and the back buffer stays put so the NEXT scene RE-RENDERS this same undisplayed
				// buffer. Rotating here was the FSKIP4 bug: it consumed buffers at tick rate against
				// a flip-rate free path — structural starvation, 204ms display-sync stalls on capture
				// scenes. Full rate + safety reasoning at vglSwapBuffersSkipFlip above.
			} else {
			struct display_queue_callback_data queue_cb_data;
			queue_cb_data.addr = gxm_color_surfaces_addr[gxm_back_buffer_index];
#ifdef HAVE_PROFILING
			tick = sceKernelGetProcessTimeLow();
#endif
			const uint32_t dq_t0 = sceKernelGetProcessTimeLow();
			sceGxmDisplayQueueAddEntry(gxm_sync_objects[gxm_front_buffer_index], gxm_sync_objects[gxm_back_buffer_index], &queue_cb_data);
			vgl_ps_display_queue_us += sceKernelGetProcessTimeLow() - dq_t0;
#ifdef HAVE_PROFILING
			gpu_stall_cnt += sceKernelGetProcessTimeLow() - tick;
#endif
#ifdef HAVE_CPU_TRACER
			sceRazorCpuSync();
#endif
			gxm_front_buffer_index = gxm_back_buffer_index;
			gxm_back_buffer_index = (gxm_back_buffer_index + 1) % gxm_display_buffer_count;
			}
		}
	}
	needs_scene_reset = GL_TRUE;
	vgl_ps_first_begin_pending = GL_TRUE;
	vgl_ps_swaps++;
	
	// Perform resolution change if there's one pending
	if (NEW_DISPLAY_WIDTH) {
		sceGxmFinish(gxm_context);
		sceGxmDestroyRenderTarget(gxm_render_target);
		if (!system_app_mode) {
			for (int i = 0; i < gxm_display_buffer_count; i++) {
				vgl_free(gxm_color_surfaces_addr[i]);
			}
		}
		DISPLAY_WIDTH = NEW_DISPLAY_WIDTH;
		DISPLAY_HEIGHT = NEW_DISPLAY_HEIGHT;
		DISPLAY_WIDTH_FLOAT = DISPLAY_WIDTH * 1.0f;
		DISPLAY_HEIGHT_FLOAT = DISPLAY_HEIGHT * 1.0f;
		DISPLAY_STRIDE = VGL_ALIGN(DISPLAY_WIDTH, 64);
		vector4f_convert_to_local_space(clear_vertices, 0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT);
		createDisplayRenderTarget();
		initDisplayColorSurfaces(GL_TRUE);
		NEW_DISPLAY_WIDTH = 0;
	}

	// Starting garbage collector job
#ifdef HAVE_SINGLE_THREADED_GC
	garbage_collector(0, NULL);
#else
	const uint32_t gc_t0 = sceKernelGetProcessTimeLow();
	sceKernelWaitSema(gc_mutex[1], 1, NULL);
	vgl_ps_gc_wait_us += sceKernelGetProcessTimeLow() - gc_t0;
	sceKernelSignalSema(gc_mutex[0], 1);
#endif
}

void vglGetPresentStats(vglPresentStats *out) {
	out->swaps = vgl_ps_swaps;
	out->end_scene_us = vgl_ps_end_scene_us;
	out->end_scenes = vgl_ps_end_scenes;
	out->capture_wait_us = vgl_efbwait_tn_us;
	out->display_queue_us = vgl_ps_display_queue_us;
	out->gc_wait_us = vgl_ps_gc_wait_us;
	out->first_begin_us = vgl_ps_first_begin_us;
	out->begin_us = vgl_ps_begin_us;
	out->begins = vgl_ps_begins;
	// sceneEnd pre-increments query_fence.value and hands the fence to sceGxmEndScene as the fragment
	// notification; the GPU writes each retired scene's value back to the address. The difference is the
	// scenes still queued or running, read without waiting on anything.
	out->scenes_in_flight = query_fence.address ? query_fence.value - *(volatile uint32_t *)query_fence.address : 0;
	out->mem_stats_walk_mask = vgl_mem_stats_walk_mask();
}

void glFinish(void) {
	// Waiting for GPU to finish drawing jobs
	sceGxmFinish(gxm_context);
}

// Mid-frame scene split: ends + submits the current DEFAULT-framebuffer scene, waits for the GPU, and
// arranges for the next draw to sceGxmBeginScene again on the SAME back buffer + depth surface (the
// swap chain is NOT advanced). After this returns, a glReadPixels on the default framebuffer reads the
// CURRENT frame content (its raw memcpy from gxm_color_surfaces_addr[gxm_back_buffer_index] otherwise
// sees stale, N-frames-old data: glFinish alone never submits the open scene's fragment job).
// Depth continuity across the split requires vglSetDisplayDepthPersistence(GL_TRUE) before vglInit*:
// the ended segment then stores depth/stencil and the restart one-shot force-loads it back. The display
// render target must have scene headroom for the extra Begin/End pairs (vglSetupDisplayRenderTarget).
// Safe no-op when no scene is active, when rendering into an FBO, or in system app mode (sceSharedFb
// owns the display scene lifecycle there).
void vglFlushFrame(void) {
	if (!needs_end_scene || in_use_framebuffer || system_app_mode)
		return;
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	if (display_depth_persist) {
		// Arm the one-shot depth reload for the restart scene (GXM samples the flag at BeginScene;
		// sceneReset disarms it right after). The scene being ended below was begun with force-store
		// enabled (initDepthStencilSurfaces), so the depth accumulated so far reaches memory at EndScene.
		sceGxmDepthStencilSurfaceSetForceLoadMode(&gxm_depth_stencil_surface, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
		display_depth_load_pending = GL_TRUE;
	}
#endif
	sceneEnd();
	needs_end_scene = GL_FALSE; // consumed: the next sceneReset must Begin, not End
	needs_scene_reset = GL_TRUE; // the next draw re-begins on the same gxm_back_buffer_index

	// PERF: a full tile flush + CPU wait for GPU idle per call (correctness-first implementation; future
	// optimization = fence/notification-based wait or a sceGxmTransferCopy readback instead).
	sceGxmFinish(gxm_context);

	// The restart resets per-scene GXM state; sceneReset's same-framebuffer path re-asserts the viewport
	// (setViewport from the x/y/z port mirrors) and the region clip (scissor_test_state), and resets the
	// DRAW_STATE_CACHE shadows (programs/textures/vstreams re-reach GXM). Default uniform buffer
	// reservations do not survive the scene change: mark uniforms dirty exactly like vglSwapBuffers does.
	dirty_frag_unifs = GL_TRUE;
	dirty_vert_unifs = GL_TRUE;
}

// Makes every display scene force-store its depth/stencil surface at scene end so vglFlushFrame scene
// splits keep depth continuity. Must be called before vglInit*. Costs one depth/stencil surface store
// per display scene. No-op (with a full-time store/load instead) when built with STORE_DEPTH_STENCIL;
// unavailable with DEPTH_STENCIL_HACK (no depth memory to store to).
void vglSetDisplayDepthPersistence(GLboolean enable) {
	display_depth_persist = enable;
}

// [EFB-ASYNC] Mid-frame scene SUBMIT with no full-pipeline wait. Identical to vglFlushFrame's scene-split
// half but WITHOUT the sceGxmFinish: it ends+submits the current DEFAULT-framebuffer scene (so the GPU has
// a fragment job queued — and query_fence fenced — for everything drawn so far) and arms the same-back-
// buffer restart, then returns. The port pairs this with vglCaptureFramebufferRegion() below, which
// notification-waits that scene's fragment fence before its transfer ([EFB-ASYNC v3]: submission order is
// NOT ordering — the transfer queue is separate from the render queue — and the display sync objects are
// the wrong fence — both device-proven by the alternating-parity stale flicker + GPU wedge of v1/v2).
// Perf context: vglFlushFrame's sceGxmFinish was the +21ms/frame device cost (modesel 27->17fps); the
// async path measured 58-62fps on the same scene.
// Same no-op guards as vglFlushFrame (no active display scene / inside an FBO / system app mode).
void vglSceneSubmit(void) {
	if (!needs_end_scene || in_use_framebuffer || system_app_mode)
		return;
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	if (display_depth_persist) {
		// Arm the one-shot depth reload for the restart scene, exactly as vglFlushFrame does.
		sceGxmDepthStencilSurfaceSetForceLoadMode(&gxm_depth_stencil_surface, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
		display_depth_load_pending = GL_TRUE;
	}
#endif
	sceneEnd();
	needs_end_scene = GL_FALSE; // consumed: the next sceneReset must Begin, not End
	needs_scene_reset = GL_TRUE; // the next draw re-begins on the same gxm_back_buffer_index
	// NO sceGxmFinish here — that is the whole point (async). Ordering against the submitted fragment job
	// is the CALLER's transfer fence ([EFB-ASYNC v2], vglCaptureFramebufferRegion's sync object), not a
	// queue-order side effect. Default uniform reservations don't survive the scene change:
	dirty_frag_unifs = GL_TRUE;
	dirty_vert_unifs = GL_TRUE;
}

// Runtime twin of vglSetDisplayDepthPersistence: switch the display depth/stencil store on or off
// between frames (GXM latches the store mode at BeginScene, so it applies from the next scene). The
// mask-update bits are initialised once, the first time it is enabled (see initDepthStencilSurfaces).
void vglSetDisplayDepthStore(GLboolean enable) {
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	static GLboolean mask_init = GL_FALSE;
	enable = enable ? GL_TRUE : GL_FALSE;
	if (enable == display_depth_persist)
		return;
	if (enable && !mask_init) {
		unsigned int w = VGL_ALIGN(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX);
		unsigned int h = VGL_ALIGN(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY);
		unsigned int samples = w * h;
		if (msaa_mode == SCE_GXM_MULTISAMPLE_2X)
			samples *= 2;
		else if (msaa_mode == SCE_GXM_MULTISAMPLE_4X)
			samples *= 4;
		vgl_memset(gxm_depth_stencil_surface.depthData, 0x80, 4 * samples);
		mask_init = GL_TRUE;
	}
	display_depth_persist = enable;
	sceGxmDepthStencilSurfaceSetForceStoreMode(&gxm_depth_stencil_surface,
		enable ? SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED : SCE_GXM_DEPTH_STENCIL_FORCE_STORE_DISABLED);
#else
	(void)enable;
#endif
}

// Scene hooks for efbcopy.c (vglEfbCopy), which lives in its own file.
int vgl_efb_display_depth_stored(void) {
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	return display_depth_persist ? 1 : 0;
#else
	return 0;
#endif
}
GLboolean needs_end_scene_get(void) { return needs_end_scene; }
// A display scene ended for a copy resumes with the depth it stored (same as vglSceneSubmit).
void vgl_efb_arm_display_depth_reload(framebuffer *src_fb) {
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	if (!src_fb && display_depth_persist) {
		sceGxmDepthStencilSurfaceSetForceLoadMode(&gxm_depth_stencil_surface, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
		display_depth_load_pending = GL_TRUE;
	}
#else
	(void)src_fb;
#endif
}
void vgl_efb_scene_end(void) {
	if (!needs_end_scene)
		return;
	sceneEnd();
	needs_end_scene = GL_FALSE;
	needs_scene_reset = GL_TRUE;
}
void vgl_efb_request_restart(void) { needs_scene_reset = GL_TRUE; }

// [EFB-ASYNC v3] EFB region capture via the transfer unit — NO CPU readback, NO glReadPixels, no
// full-pipeline stall. Call ONLY after vglSceneSubmit(). Race-free BY CONSTRUCTION with two explicit
// notification fences (see the fence comments at the waits below):
//   RAW: sceGxmNotificationWait(query_fence) — the split scene's fragment job has RETIRED before the
//        transfer is even enqueued, so the source surface is provably complete (no flicker possible);
//   WAR/consumer: the transfer signals capture_tn, and sceneEnd() waits it before submitting ANY later
//        fragment job — nothing can overwrite the source or sample the dst while the transfer runs.
// Both waits are instrumented (vgl_efbwait_fn_*/vgl_efbwait_tn_*, vitaGL.h) — the FN wait is scoped to
// fragment jobs up to the split (not a sceGxmFinish pipeline drain); TN is ~always pre-signaled. The dst
// GL texture MUST already have RGBA8 linear storage of at least w*h (the port glTexImage2D's it first).
//   * src = gxm_color_surfaces_addr[gxm_back_buffer_index], linear RGBA8, stride DISPLAY_STRIDE*4 bytes.
//   * src_y_gl is GL bottom-up (0 = bottom row); the GC/GL capture wants top-down texel rows. The Y-flip is
//     absorbed entirely by the SOURCE origin (top-down row = DISPLAY_HEIGHT-(src_y_gl+h)); rows are then
//     read and written downward with positive strides — byte-identical to the CPU path (glReadPixels'
//     bottom-up read + pc_gx_efb_flip_remap's row swap). No channel remap: both ends are
//     U8U8U8U8_ABGR; the R8/RA8/RGB565 "remap" was only channel replication (sampled values identical) and
//     is applied by the port's TEV resolve/swizzle at sample time, not here.
// Returns 0 on success, <0 if the rect/texture is unusable (caller falls back to the legacy readback path).
int vglCaptureFramebufferRegion(GLuint dst_gl_tex, int src_x, int src_y_gl, int w, int h) {
	if (w <= 0 || h <= 0 || dst_gl_tex == 0 || dst_gl_tex >= TEXTURES_NUM)
		return -1;
	if (in_use_framebuffer || system_app_mode)
		return -2; // only the default display surface is a valid transfer source here
	texture *tex = &texture_slots[dst_gl_tex];
	if (tex->status != TEX_VALID || !tex->data)
		return -3;
	void *dst_base = sceGxmTextureGetData(&tex->gxm_tex);
	if (!dst_base)
		return -4;
	// vitaGL linear textures allocate rows at VGL_ALIGN(w,8)*bpp (gpu_alloc_texture). The EFB captures are
	// RGBA8 (bpp=4); dst row stride = VGL_ALIGN(dst_w,8)*4. Use the texture's OWN width for the stride so a
	// pooled texture that is wider than the copy rect still writes correctly.
	uint32_t dst_w = sceGxmTextureGetWidth(&tex->gxm_tex);
	int dst_stride = (int)(VGL_ALIGN(dst_w, 8) * 4);
	// src: linear RGBA8 display back buffer. GL rows are bottom-up (src_y_gl=0 is the BOTTOM screen row);
	// the capture wants top-down texel rows. Convert the GL rect origin to a top-down source ROW: the rect's
	// TOP screen row (which becomes texture row 0) sits at top-down source index DISPLAY_HEIGHT-(src_y_gl+h).
	// The Y-flip is therefore fully absorbed by this src origin — we then read rows downward and write them
	// downward (POSITIVE dest stride) so texture-row-0 = region-top, byte-identical to the CPU path
	// (glReadPixels' bottom-up read + pc_gx_efb_flip_remap's row swap yields the same top-down texture).
	uint8_t *src_base = (uint8_t *)gxm_color_surfaces_addr[gxm_back_buffer_index];
	int src_stride = DISPLAY_STRIDE * 4;
	int src_top_row = DISPLAY_HEIGHT - (src_y_gl + h); // top row of the rect in top-down source space
	if (src_top_row < 0 || src_x < 0 || src_x + w > DISPLAY_STRIDE || src_top_row + h > DISPLAY_HEIGHT)
		return -5;
	// [EFB-ASYNC v3] RAW fence: CPU-wait the fork's OWN per-scene fragment fence before enqueuing the
	// transfer. sceneEnd passes query_fence as sceGxmEndScene's fragmentNotification (value pre-incremented
	// per scene; the GPU writes value to address when that scene's fragment job RETIRES — the exact
	// mechanism vitaGL's occlusion queries already wait on, buffers.c:168). vglSceneSubmit() just ended the
	// split scene, so the current {address,value} identifies precisely it: sceGxmNotificationWait returns
	// the moment the capture source is complete. NOT a sceGxmFinish-class stall — it drains fragment jobs
	// up to the split only (no vertex/transfer/display drain, no pipeline teardown); instrumented
	// (vgl_efbwait_fn_*) so the device run attributes the real cost.
	// v2 POST-MORTEM (device-proven): fencing the transfer on gxm_sync_objects[back] +
	// SCE_GXM_TRANSFER_FRAGMENT_SYNC did NOT order it — identical alternating-parity flicker + GPU wedge
	// as unfenced v1. Those sync objects gate the DISPLAY lifecycle (sceGxmDisplayQueueAddEntry pairs
	// front/back at swap; per-swapchain-buffer epoch = the parity signature), not intra-frame fragment
	// completion. Explicit notifications are the correct primitive.
	// [EFB-ASYNC v3.1] backlog depth before the wait: value - *address = scenes the GPU still owes
	// (address holds the LAST RETIRED scene's value). See vgl_efbwait_bk decl for what 1 vs 2+ means.
	{
		uint32_t bk_done = *(volatile uint32_t *)query_fence.address;
		uint32_t bk = query_fence.value - bk_done;
		vgl_efbwait_bk[bk > 3u ? 3u : bk]++;
	}
	uint64_t fn_t0 = sceKernelGetProcessTimeWide();
	// [EFB-ASYNC v3.2] poll instead of sceGxmNotificationWait so the wait can be SPLIT at the moment all
	// OLDER scenes (value-1) retire — see vgl_efbwait_prev_us/own_us decl. Same completion condition.
	{
		volatile uint32_t *fence_addr = (volatile uint32_t *)query_fence.address;
		uint32_t target = query_fence.value;
		uint64_t prev_mark = fn_t0;
		GLboolean prev_done = (*fence_addr + 1u == target) ? GL_TRUE : GL_FALSE; // older scenes already retired
		while (*fence_addr != target) {
			if (!prev_done && *fence_addr + 1u == target) {
				prev_done = GL_TRUE;
				prev_mark = sceKernelGetProcessTimeWide();
				vgl_efbwait_prev_us += (uint32_t)(prev_mark - fn_t0);
			}
			sceKernelDelayThread(100);
		}
		if (!prev_done) // skipped straight past value-1 between polls: attribute the whole wait to prev
			vgl_efbwait_prev_us += (uint32_t)(sceKernelGetProcessTimeWide() - fn_t0);
		else
			vgl_efbwait_own_us += (uint32_t)(sceKernelGetProcessTimeWide() - prev_mark);
	}
	uint32_t fn_dt = (uint32_t)(sceKernelGetProcessTimeWide() - fn_t0);
	vgl_efbwait_fn_us += fn_dt;
	vgl_efbwait_fn_cnt++;
	if (fn_dt > vgl_efbwait_fn_max)
		vgl_efbwait_fn_max = fn_dt;
	// Transfer-done notification (the WAR/consumer fence sceneEnd waits on before submitting the next
	// fragment job): notification-region slot 1, monotonic value. Back-to-back captures without a scene
	// end between them just bump the value again — the transfer queue is FIFO, so waiting for the LATEST
	// value in sceneEnd covers every outstanding capture transfer.
	if (!capture_tn.address) {
		capture_tn.address = sceGxmGetNotificationRegion() + 1;
		*capture_tn.address = 0;
	}
	capture_tn.value++;
	int r = sceGxmTransferCopy(
		w, h, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE,
		SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR, SCE_GXM_TRANSFER_LINEAR,
		src_base, (uint32_t)src_x, (uint32_t)src_top_row, src_stride,
		SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR, SCE_GXM_TRANSFER_LINEAR,
		dst_base, 0, 0, dst_stride,
		NULL, 0, &capture_tn);
	if (r) {
		capture_tn.value--; // NOT enqueued: leaving the bumped value would make sceneEnd's wait hang forever
		return r;
	}
	capture_tn_pending = GL_TRUE;
	return 0;
}

// [FBORT] Offscreen-FBO region capture — the FBO analog of vglCaptureFramebufferRegion, for the MP4 port's
// bit27 render-to-FBO frame mode (port gx_gl.cpp). vglCaptureFramebufferRegion rejects an FBO source
// (in_use_framebuffer != NULL -> returns -2) because it sources the DISPLAY back buffer; but the whole point
// of bit27 is that GXCopyTex (shadow maps / see-through water / stamp feedback) must capture the OFFSCREEN
// scene the frame renders into, with a fence that retires at GPU speed (~ms) instead of being coupled to the
// display flip cadence (the display-scene split's fragment fence sits behind display-queue backpressure —
// device-measured 200ms/frame on the board-select transition; §"Why" in the port). This helper ends the
// IN-USE FBO scene, waits its fragment fence, transfers the region from the FBO color surface into the dst
// GL texture, then arms a same-FBO restart — the identical two-fence discipline as the display capture
// (RAW query_fence before the transfer + capture_tn WAR fence consumed in the next sceneEnd), so it is
// race-free by the same construction the display path proved on device (v1 unfenced / v2 display-sync-fenced
// both wedged the GPU — see the vglCaptureFramebufferRegion v2 post-mortem).
//
// DEPTH CONTINUITY across the split: an FBO depth surface is NOT force-store/force-load managed by default
// (display_depth_persist is display-only; framebuffers.c:863 shows the per-FBO force-store idiom). A bare
// sceneEnd would DISCARD the depth accumulated so far (tiled-GPU behavior), so geometry drawn AFTER the
// capture would z-test against a cleared depth buffer (MP4 draws opaque world, captures, then draws more
// world every EFB frame). We therefore force-STORE the FBO depth at this EndScene and one-shot force-LOAD it
// at the restart BeginScene (sceneReset), mirroring the display depth-persistence one-shot exactly.
//
// Y-ORIENTATION: unlike the display back buffer (which glReadPixels/transfer read bottom-up), a vitaGL FBO
// color texture without HAVE_UNFLIPPED_FBOS is stored TOP-DOWN already at its data pointer (framebuffers.c
// glReadPixels FBO branch reads fb->data with a POSITIVE row advance). The GL rect the port hands us is
// bottom-up (src_y_gl=0 = bottom row); the top screen row of the rect (texture row 0) therefore sits at
// top-down source row (fb->height - (src_y_gl + h)). Same source-origin Y-flip as the display path, so the
// captured texture rows come out top-down = byte-identical sampling to the display capture.
// [CAPCOALESCE] shared body for the split and NO-SPLIT variants. split=GL_TRUE = the original
// vglCaptureFboRegion behavior (end the FBO scene + fence it + transfer + arm restart). split=GL_FALSE =
// TRANSFER-ONLY: no scene end, no depth store/load arming — just the (pre-satisfied) fence check and the DMA.
//
// FENCE-VALIDITY INVARIANT for split=GL_FALSE (the caller MUST guarantee it; the MP4 port's g_vrt_surf_epoch
// tracks it): every fragment job that wrote the FBO COLOR surface has retired, and no COLOR-writing work has
// been issued since. Precisely:
//   * Fragment jobs are created only at sceGxmEndScene (sceneEnd, all call sites). The caller's previous
//     SPLIT capture FN-waited query_fence AFTER its sceneEnd, proving retirement of every fragment job
//     submitted up to that point.
//   * If no draw/clear/blit executed since that wait, no NEW color-writing fragment job was submitted OR
//     issued — so the color surface bytes are exactly the fenced content. NOTE the load-bearing subtlety: the
//     currently-OPEN scene may be non-empty with STENCIL-ONLY work (vitaGL's update_scissor_test draws
//     stencil-mask quads on every glScissor — the MP4 post-capture reassert emits one per capture). Those
//     quads never write color, and their fragment job is not even submitted until a later sceneEnd — which
//     TN-waits capture_tn BEFORE submitting, so it cannot race this transfer either (WAR fence, unchanged).
//   * The FN wait below is retained as defense-in-depth: it waits the CURRENT query_fence value, which per
//     the invariant is already signaled (~0 cost). If a scene end somehow slipped in (invariant violation),
//     the wait still orders the transfer after every SUBMITTED fragment job — the only unprotected case would
//     be color draws sitting in a still-open scene, which the caller's epoch tracking exists to exclude.
// Back-to-back no-split transfers are FIFO in the transfer queue and each bumps capture_tn; sceneEnd waits
// the LATEST value, covering all of them (same contract as back-to-back display captures).
static int capture_fbo_common(GLuint dst_gl_tex, int src_x, int src_y_gl, int w, int h, GLboolean split) {
	if (w <= 0 || h <= 0 || dst_gl_tex == 0 || dst_gl_tex >= TEXTURES_NUM)
		return -1;
	if (!in_use_framebuffer || system_app_mode)
		return -2; // only valid while an FBO scene is the render target (the bit27 inverse of the display capture)
	if (split && !needs_end_scene)
		return -6; // split requested but no open FBO scene (nothing drawn yet this frame) — caller falls back
	framebuffer *fb = in_use_framebuffer;
	if (!fb->data || !fb->tex)
		return -7;
	texture *tex = &texture_slots[dst_gl_tex];
	if (tex->status != TEX_VALID || !tex->data)
		return -3;
	void *dst_base = sceGxmTextureGetData(&tex->gxm_tex);
	if (!dst_base)
		return -4;
	// Source rect bounds check (top-down source space; see Y-ORIENTATION above).
	int src_top_row = fb->height - (src_y_gl + h);
	int src_px_stride = fb->stride / 4; // fb->stride is bytes; RGBA8 -> 4 bpp. Row pixel stride for bounds.
	if (src_top_row < 0 || src_x < 0 || src_x + w > src_px_stride || src_top_row + h > fb->height)
		return -5;
	uint32_t dst_w = sceGxmTextureGetWidth(&tex->gxm_tex);
	int dst_stride = (int)(VGL_ALIGN(dst_w, 8) * 4);

	if (split) {
	// [FBORT] force-STORE the FBO depth at the scene end below so the restart can resume it (depth continuity
	// across the mid-frame capture split). GXM latches the mode at EndScene, so arm it before sceneEnd().
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	if (fb->depthbuffer_ptr)
		sceGxmDepthStencilSurfaceSetForceStoreMode(fb->depthbuffer_ptr, SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);
#endif

	// End the FBO scene: submits its fragment job with query_fence as the fragment-retire notification
	// (sceneEnd itself bumps query_fence.value — the earlier extra pre-bump here only skewed the bk
	// histogram and was removed with [CAPCOALESCE]). This is the FBO twin of vglSceneSubmit (which no-ops
	// inside an FBO). We keep active_write_fb == fb, so the next draw's sceneReset re-Begins on the SAME
	// FBO surface (needs_scene_reset forces it even though the pointer is unchanged).
	sceneEnd();
	needs_end_scene = GL_FALSE;   // consumed: the next sceneReset must Begin, not End
	needs_scene_reset = GL_TRUE;  // re-Begin the same FBO on the next draw
	dirty_frag_unifs = GL_TRUE;   // default uniform reservations don't survive the scene change
	dirty_vert_unifs = GL_TRUE;
#if !defined(STORE_DEPTH_STENCIL) && !defined(DEPTH_STENCIL_HACK)
	// Arm the one-shot force-LOAD for the FBO restart (consumed in sceneReset). Store stays permanently armed
	// on this FBO's surface for the rest of the frame (subsequent splits keep continuity); the frame-start
	// clear rebuilds depth next frame regardless.
	if (fb->depthbuffer_ptr) {
		sceGxmDepthStencilSurfaceSetForceLoadMode(fb->depthbuffer_ptr, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
		fbo_depth_load_pending = GL_TRUE;
		fbo_depth_load_fb = fb;
	}
#endif
	} // split

	// [FBORT] RAW fence: wait the FBO scene's fragment job retirement before the transfer reads its color
	// surface (glReadPixels-of-FBO / transfer-from-FBO have NO implicit render-queue ordering — framebuffers.c
	// glReadPixels is a raw memcpy of fb->data). Same poll+instrument as the display capture so the port's
	// [efb-wait] meters stay meaningful; the FBO fence is pure GPU latency (no display gating). On the
	// [CAPCOALESCE] no-split path this waits the PREVIOUS split's (already-signaled) fence — defense in depth,
	// ~0 cost (see the invariant block above).
	{
		uint32_t bk_done = *(volatile uint32_t *)query_fence.address;
		uint32_t bk = query_fence.value - bk_done;
		vgl_efbwait_bk[bk > 3u ? 3u : bk]++;
	}
	uint64_t fn_t0 = sceKernelGetProcessTimeWide();
	{
		volatile uint32_t *fence_addr = (volatile uint32_t *)query_fence.address;
		uint32_t target = query_fence.value;
		uint64_t prev_mark = fn_t0;
		GLboolean prev_done = (*fence_addr + 1u == target) ? GL_TRUE : GL_FALSE;
		while (*fence_addr != target) {
			if (!prev_done && *fence_addr + 1u == target) {
				prev_done = GL_TRUE;
				prev_mark = sceKernelGetProcessTimeWide();
				vgl_efbwait_prev_us += (uint32_t)(prev_mark - fn_t0);
			}
			sceKernelDelayThread(100);
		}
		if (!prev_done)
			vgl_efbwait_prev_us += (uint32_t)(sceKernelGetProcessTimeWide() - fn_t0);
		else
			vgl_efbwait_own_us += (uint32_t)(sceKernelGetProcessTimeWide() - prev_mark);
	}
	uint32_t fn_dt = (uint32_t)(sceKernelGetProcessTimeWide() - fn_t0);
	vgl_efbwait_fn_us += fn_dt;
	vgl_efbwait_fn_cnt++;
	if (fn_dt > vgl_efbwait_fn_max)
		vgl_efbwait_fn_max = fn_dt;

	// Transfer the region from the FBO color surface (linear RGBA8, fb->stride bytes) into the dst texture.
	// The capture_tn WAR fence is consumed by the NEXT sceneEnd (the restart scene / copy-clear / the open
	// scene's eventual end on the no-split path) before it submits any fragment job — nothing overwrites the
	// FBO color or samples the dst mid-transfer.
	if (!capture_tn.address) {
		capture_tn.address = sceGxmGetNotificationRegion() + 1;
		*capture_tn.address = 0;
	}
	capture_tn.value++;
	int r = sceGxmTransferCopy(
		w, h, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE,
		SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR, SCE_GXM_TRANSFER_LINEAR,
		fb->data, (uint32_t)src_x, (uint32_t)src_top_row, fb->stride,
		SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR, SCE_GXM_TRANSFER_LINEAR,
		dst_base, 0, 0, dst_stride,
		NULL, 0, &capture_tn);
	if (r) {
		capture_tn.value--; // NOT enqueued: leaving the bumped value would hang the next sceneEnd's TN wait
		return r;
	}
	capture_tn_pending = GL_TRUE;
	return 0;
}
int vglCaptureFboRegion(GLuint dst_gl_tex, int src_x, int src_y_gl, int w, int h) {
	return capture_fbo_common(dst_gl_tex, src_x, src_y_gl, w, h, GL_TRUE);
}
// [CAPCOALESCE] transfer-only FBO capture: NO scene end, NO fence creation, NO depth store/load arming. For
// consecutive captures with no intervening color-writing work (the strip/mosaic WIPE class: dozens of
// GXCopyTex sub-rects per frame with only stencil-only scissor updates between them), the previous split's
// completed FN wait already proves the color surface content is final — each additional capture is then a
// bare DMA enqueue. The caller (MP4 pc_gx_efb_copy_exec_fbo) enforces the fence-validity invariant documented
// at capture_fbo_common via its RT-side surface-write epoch. Device motive: 0.69fps board-select wipe =
// ~75 captures/tick x ~10ms full split each (empty-scene end + ~MB-scale depth/color store/load + fence);
// coalesced = 1 split + 74 DMA enqueues.
int vglCaptureFboRegionNoSplit(GLuint dst_gl_tex, int src_x, int src_y_gl, int w, int h) {
	return capture_fbo_common(dst_gl_tex, src_x, src_y_gl, w, h, GL_FALSE);
}

// [XFERPRESENT] Present the in-use FBO via the TRANSFER ENGINE — no display scene, no display fragment
// job, EVER. Device forensics (MP4 bit27 wipe, 2026-07-07): the GL-blit present is a fullscreen DRAW on
// a display scene whose fragment job is display-sync-gated (sceneReset BeginScene with
// gxm_sync_objects[back]); every later scene's EndScene then blocks on job/param-buffer recycling held
// behind that gated job (measured 1.85-1.89s inside vglCaptureFboRegion with the FN/TN fences at
// 7ms/3us — the block predates the fence, inside sceGxmEndScene). Removing the display scene entirely
// removes the only display-gated job from the render queue, so FBO scene retirement becomes pure GPU
// latency everywhere. Sequence: end the FBO scene (submits its job), CPU-wait its retirement (bounded
// by real GPU work now), DMA the FBO color to the display back buffer, CPU-wait the DMA (~1ms for 2MB)
// so the following flip cannot scan out a half-written buffer. Y: fb->data and the display surface are
// both top-down in memory (capture_fbo_common's src_top_row math is the precedent) -> straight copy.
int vglPresentFboTransfer(void) {
	if (!in_use_framebuffer || system_app_mode)
		return -2;
	framebuffer *fb = in_use_framebuffer;
	if (!fb->data)
		return -7;
	if (needs_end_scene) {
		sceneEnd();                     // submits the FBO fragment job; TN-waits any pending capture transfer
		needs_end_scene = GL_FALSE;
		needs_scene_reset = GL_TRUE;    // next frame's first draw re-begins (the port re-binds the FBO then)
		dirty_frag_unifs = GL_TRUE;
		dirty_vert_unifs = GL_TRUE;
	}
	sceGxmNotificationWait(&query_fence);   // RAW: FBO job retired -> fb->data is the final frame
	if (!capture_tn.address) {
		capture_tn.address = sceGxmGetNotificationRegion() + 1;
		*capture_tn.address = 0;
	}
	capture_tn.value++;
	int r = sceGxmTransferCopy(
		fb->width, fb->height, 0, 0, SCE_GXM_TRANSFER_COLORKEY_NONE,
		SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR, SCE_GXM_TRANSFER_LINEAR,
		fb->data, 0, 0, fb->stride,
		SCE_GXM_TRANSFER_FORMAT_U8U8U8U8_ABGR, SCE_GXM_TRANSFER_LINEAR,
		gxm_color_surfaces_addr[gxm_back_buffer_index], 0, 0, DISPLAY_STRIDE * 4,
		NULL, 0, &capture_tn);
	if (r) {
		capture_tn.value--;   // not enqueued: a stale bumped value would hang the next TN wait forever
		return r;
	}
	sceGxmNotificationWait(&capture_tn);
	capture_tn_pending = GL_FALSE;   // already waited here; sceneEnd need not re-wait
	// CRITICAL state handoff (device-paid: 212Hz free-run with a STATIC screen on first ship): the
	// vglSwapBuffers flip block is gated on !in_use_framebuffer (gxm.c ~1002). The old GL-blit present
	// re-pointed it as a side effect of the blit DRAW's sceneReset onto the default framebuffer; the
	// transfer present has no draw, so hand the state over explicitly or NO FLIP IS EVER QUEUED.
	// needs_scene_reset stays TRUE (set above), which also skips vglSwapBuffers' pre-flip sceneEnd()
	// (gxm.c ~982) — the FBO scene is already ended and fenced here.
	in_use_framebuffer = NULL;
	return 0;
}

void glReleaseShaderCompiler(void) {
	if (is_shark_online) {
		shark_end();
		is_shark_online = GL_FALSE;
	}
}

void glFlush(void) {
}

void vglSetDisplayCallback(void (*cb)(void *framebuf)) {
	vgl_display_cb = cb;
}

void vglSetupShaderPatcher(uint32_t buffer_mem_size, uint32_t vertex_usse_mem_size, uint32_t fragment_usse_mem_size) {
	shader_patcher_buffer_size = buffer_mem_size;
	shader_patcher_vertex_usse_size = vertex_usse_mem_size;
	shader_patcher_fragment_usse_size = fragment_usse_mem_size;
}

void vglSetupDisplayRenderTarget(uint8_t size) {
#ifndef SKIP_ERROR_HANDLING
	if (size > MAX_SCENES_PER_FRAME) {
		SET_GL_ERROR_WITH_VALUE(GL_INVALID_VALUE, size)
	}
#endif
	gxm_display_rt_size = size;
}