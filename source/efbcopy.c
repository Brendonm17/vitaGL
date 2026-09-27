/*
 * efbcopy.c: GPU-side copies of the current render target into a texture
 * (the GameCube EFB-to-texture copy, GXCopyTex), for Open Nectar.
 *
 * The copy is a render pass, not a transfer: the scene being drawn is ended,
 * a tiny scene on the destination framebuffer draws one quad that samples the
 * ended surface (colour, or depth when the display keeps it), and the next GL
 * draw resumes the original target. Scenes are consumed by the GPU in order,
 * so there is no CPU wait anywhere -- unlike vglCaptureFramebufferRegion,
 * whose transfer unit is not ordered against the render queue and has to be
 * fenced from the CPU.
 *
 * The quad is drawn with raw GXM on programs the caller hands over once
 * (vglEfbCopySetPrograms). Nothing GL-visible is touched: the destination
 * scene is begun and ended here, and the resumed scene goes through
 * sceneReset's framebuffer-change path, which re-applies the GL viewport,
 * scissor and cull state and invalidates the draw-state caches.
 */
#include "shared.h"

#define EFBCOPY_MAX_FP 8

static SceGxmShaderPatcherId efb_vp_id;
static SceGxmShaderPatcherId efb_fp_id[EFBCOPY_MAX_FP];
static const SceGxmProgram *efb_vp_gxp = NULL;
static SceGxmVertexProgram *efb_vprog = NULL;
static SceGxmFragmentProgram *efb_fprog[EFBCOPY_MAX_FP];
static int efb_nfp = 0;

// Scene-machinery hooks (gxm.c owns the scene state).
int vgl_efb_display_depth_stored(void);
GLboolean needs_end_scene_get(void);
void vgl_efb_arm_display_depth_reload(framebuffer *src_fb);
void vgl_efb_scene_end(void);
void vgl_efb_request_restart(void);

int vglEfbCopySetPrograms(const SceGxmProgram *vp, const SceGxmProgram *const *fps, int nfp) {
	if (!vp || !fps || nfp <= 0 || nfp > EFBCOPY_MAX_FP)
		return -1;
	if (sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, vp, &efb_vp_id) < 0)
		return -2;
	const SceGxmProgramParameter *p_pos = sceGxmProgramFindParameterByName(vp, "aPos");
	const SceGxmProgramParameter *p_uv0 = sceGxmProgramFindParameterByName(vp, "aUV01");
	const SceGxmProgramParameter *p_uv1 = sceGxmProgramFindParameterByName(vp, "aUV23");
	if (!p_pos || !p_uv0 || !p_uv1)
		return -3;
	SceGxmVertexAttribute attrs[3];
	attrs[0].streamIndex = 0;
	attrs[0].offset = 0;
	attrs[0].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attrs[0].componentCount = 2;
	attrs[0].regIndex = sceGxmProgramParameterGetResourceIndex(p_pos);
	attrs[1].streamIndex = 0;
	attrs[1].offset = 8;
	attrs[1].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attrs[1].componentCount = 4;
	attrs[1].regIndex = sceGxmProgramParameterGetResourceIndex(p_uv0);
	attrs[2].streamIndex = 0;
	attrs[2].offset = 24;
	attrs[2].format = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attrs[2].componentCount = 4;
	attrs[2].regIndex = sceGxmProgramParameterGetResourceIndex(p_uv1);
	SceGxmVertexStream stream;
	stream.stride = 40;
	stream.indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;
	if (sceGxmShaderPatcherCreateVertexProgram(gxm_shader_patcher, efb_vp_id, attrs, 3, &stream, 1, &efb_vprog) < 0)
		return -4;
	for (int i = 0; i < nfp; i++) {
		if (sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, fps[i], &efb_fp_id[i]) < 0)
			return -5;
		if (sceGxmShaderPatcherCreateFragmentProgram(gxm_shader_patcher, efb_fp_id[i], SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
				SCE_GXM_MULTISAMPLE_NONE, NULL, vp, &efb_fprog[i]) < 0)
			return -6;
	}
	efb_vp_gxp = vp;
	efb_nfp = nfp;
	return 0;
}

// src_depth: 0 = colour of the current target, 1 = its depth (display only, and only
// when vglSetDisplayDepthPersistence(GL_TRUE) keeps display depth in memory).
// u0,v0 (top-left) .. u1,v1 (bottom-right): the source rect in the surface's own
// texture space, v = 0 at the top row. box: four taps a quarter of a destination
// texel apart (the GX half-scale box filter), else one tap.
int vglEfbCopy(GLuint dst_fbo, int fp_index, int src_depth, float u0, float v0, float u1, float v1, int box) {
	framebuffer *dst = (framebuffer *)dst_fbo;
	if (!efb_vprog || fp_index < 0 || fp_index >= efb_nfp || !dst || !dst->tex || system_app_mode)
		return -1;
	framebuffer *src_fb = in_use_framebuffer;
	if (src_fb == dst)
		return -2;

	// The source surface, as a texture.
	SceGxmTexture src;
	if (src_depth) {
		if (src_fb) {
			if (!src_fb->depthbuffer_ptr)
				return -3;
			return -3; // FBO depth is not stored (not needed by the game)
		}
		if (!vgl_efb_display_depth_stored())
			return -4;
		sceGxmTextureInitLinear(&src, gxm_depth_stencil_surface.depthData, SCE_GXM_TEXTURE_FORMAT_DF32M,
			VGL_ALIGN(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX), VGL_ALIGN(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY), 0);
		sceGxmTextureSetMinFilter(&src, SCE_GXM_TEXTURE_FILTER_POINT);
		sceGxmTextureSetMagFilter(&src, SCE_GXM_TEXTURE_FILTER_POINT);
		// the surface is wider/taller than the display only by tile padding: rescale the rect
		const float sx = (float)DISPLAY_WIDTH / (float)VGL_ALIGN(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX);
		const float sy = (float)DISPLAY_HEIGHT / (float)VGL_ALIGN(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY);
		u0 *= sx; u1 *= sx; v0 *= sy; v1 *= sy;
	} else if (src_fb) {
		src = src_fb->tex->gxm_tex;
		sceGxmTextureSetMinFilter(&src, SCE_GXM_TEXTURE_FILTER_LINEAR);
		sceGxmTextureSetMagFilter(&src, SCE_GXM_TEXTURE_FILTER_LINEAR);
	} else {
		sceGxmTextureInitLinearStrided(&src, gxm_color_surfaces_addr[gxm_back_buffer_index],
			SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_STRIDE * 4);
		sceGxmTextureSetMinFilter(&src, SCE_GXM_TEXTURE_FILTER_LINEAR);
		sceGxmTextureSetMagFilter(&src, SCE_GXM_TEXTURE_FILTER_LINEAR);
	}
	sceGxmTextureSetUAddrMode(&src, SCE_GXM_TEXTURE_ADDR_CLAMP);
	sceGxmTextureSetVAddrMode(&src, SCE_GXM_TEXTURE_ADDR_CLAMP);

	// End the scene that owns the source so its fragment job (and, for the
	// display, the depth store) is submitted before the copy scene.
	framebuffer *saved_write = active_write_fb;
	if (needs_end_scene_get()) {
		vgl_efb_arm_display_depth_reload(src_fb);
		vgl_efb_scene_end();
	}

	// The copy scene on the destination.
	active_write_fb = dst;
	sceneReset();
	const int w = dst->width, h = dst->height;
	sceGxmSetViewport(gxm_context, (float)w * 0.5f, (float)w * 0.5f, (float)h * 0.5f, -(float)h * 0.5f, 0.5f, 0.5f);
	sceGxmSetRegionClip(gxm_context, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, w - 1, h - 1);
	sceGxmSetCullMode(gxm_context, SCE_GXM_CULL_NONE);
	sceGxmSetFrontPolygonMode(gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
	sceGxmSetBackPolygonMode(gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
	sceGxmSetFrontDepthFunc(gxm_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetBackDepthFunc(gxm_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(gxm_context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetBackDepthWriteEnable(gxm_context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetFrontStencilFunc(gxm_context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	sceGxmSetBackStencilFunc(gxm_context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	sceGxmSetVertexProgram(gxm_context, efb_vprog);
	sceGxmSetFragmentProgram(gxm_context, efb_fprog[fp_index]);
	sceGxmSetFragmentTexture(gxm_context, 0, &src);

	// Four taps a quarter of a destination texel from its centre.
	const float qu = box ? (u1 - u0) / (float)w * 0.25f : 0.0f;
	const float qv = box ? (v1 - v0) / (float)h * 0.25f : 0.0f;
	float *vtx = (float *)gpu_alloc_mapped_temp(4 * 10 * sizeof(float));
	uint16_t *idx = (uint16_t *)gpu_alloc_mapped_temp(4 * sizeof(uint16_t));
	if (!vtx || !idx)
		goto done;
	// strip order: top-left, bottom-left, top-right, bottom-right (NDC y up = row 0)
	const float px[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
	const float py[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
	const float pu[4] = { u0, u0, u1, u1 };
	const float pv[4] = { v0, v1, v0, v1 };
	for (int i = 0; i < 4; i++) {
		float *o = vtx + i * 10;
		o[0] = px[i];
		o[1] = py[i];
		o[2] = pu[i] - qu; o[3] = pv[i] - qv;
		o[4] = pu[i] + qu; o[5] = pv[i] - qv;
		o[6] = pu[i] - qu; o[7] = pv[i] + qv;
		o[8] = pu[i] + qu; o[9] = pv[i] + qv;
		idx[i] = (uint16_t)i;
	}
	sceGxmSetVertexStream(gxm_context, 0, vtx);
	sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_INDEX_FORMAT_U16, idx, 4);

done:
	// Close the copy scene and resume the original target at the next draw.
	vgl_efb_scene_end();
	active_write_fb = saved_write;
	// The copy scene is closed; the source target owns the next scene. The swap
	// only flips when in_use_framebuffer is the display, so a frame whose last
	// operation is a copy must still name the display here.
	in_use_framebuffer = saved_write;
	vgl_efb_request_restart();
	dirty_frag_unifs = GL_TRUE;
	dirty_vert_unifs = GL_TRUE;
	return 0;
}

// The GX display copy's vertical filter, applied to the finished frame: the
// display is copied into tmp_fbo's texture, then drawn back over itself with
// rows y-1, y, y+1 weighted edge/centre/edge. row_dv is one EFB line in the
// display's texture space. Call right before the swap.
int vglEfbFilterDisplay(GLuint tmp_fbo, int fp_copy, int fp_filter, float row_dv, float w_edge, float w_centre) {
	framebuffer *tmp = (framebuffer *)tmp_fbo;
	if (!efb_vprog || !tmp || !tmp->tex || fp_filter < 0 || fp_filter >= efb_nfp || in_use_framebuffer || active_write_fb)
		return -1;
	int r = vglEfbCopy(tmp_fbo, fp_copy, 0, 0.0f, 0.0f, 1.0f, 1.0f, 0);
	if (r < 0)
		return r;
	// The display again (the same back buffer), drawn over entirely.
	sceneReset();
	const float w = (float)DISPLAY_WIDTH, h = (float)DISPLAY_HEIGHT;
	sceGxmSetViewport(gxm_context, w * 0.5f, w * 0.5f, h * 0.5f, -h * 0.5f, 0.5f, 0.5f);
	sceGxmSetRegionClip(gxm_context, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1);
	sceGxmSetCullMode(gxm_context, SCE_GXM_CULL_NONE);
	sceGxmSetFrontPolygonMode(gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
	sceGxmSetBackPolygonMode(gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
	sceGxmSetFrontDepthFunc(gxm_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetBackDepthFunc(gxm_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(gxm_context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetBackDepthWriteEnable(gxm_context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetFrontStencilFunc(gxm_context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	sceGxmSetBackStencilFunc(gxm_context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	SceGxmTexture src = tmp->tex->gxm_tex;
	sceGxmTextureSetMinFilter(&src, SCE_GXM_TEXTURE_FILTER_LINEAR);
	sceGxmTextureSetMagFilter(&src, SCE_GXM_TEXTURE_FILTER_LINEAR);
	sceGxmTextureSetUAddrMode(&src, SCE_GXM_TEXTURE_ADDR_CLAMP);
	sceGxmTextureSetVAddrMode(&src, SCE_GXM_TEXTURE_ADDR_CLAMP);
	sceGxmSetVertexProgram(gxm_context, efb_vprog);
	sceGxmSetFragmentProgram(gxm_context, efb_fprog[fp_filter]);
	sceGxmSetFragmentTexture(gxm_context, 0, &src);
	float *vtx = (float *)gpu_alloc_mapped_temp(4 * 10 * sizeof(float));
	uint16_t *idx = (uint16_t *)gpu_alloc_mapped_temp(4 * sizeof(uint16_t));
	if (vtx && idx) {
		const float px[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
		const float py[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
		const float pu[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
		const float pv[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
		for (int i = 0; i < 4; i++) {
			float *o = vtx + i * 10;
			o[0] = px[i]; o[1] = py[i];
			o[2] = pu[i]; o[3] = pv[i] - row_dv;   // row above
			o[4] = pu[i]; o[5] = pv[i];            // this row
			o[6] = pu[i]; o[7] = pv[i] + row_dv;   // row below
			o[8] = w_edge; o[9] = w_centre;
			idx[i] = (uint16_t)i;
		}
		sceGxmSetVertexStream(gxm_context, 0, vtx);
		sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_INDEX_FORMAT_U16, idx, 4);
	}
	vgl_efb_scene_end();
	vgl_efb_request_restart();
	dirty_frag_unifs = GL_TRUE;
	dirty_vert_unifs = GL_TRUE;
	return 0;
}

// A complete mip chain supplied by the caller: level l is (w >> l) x (h >> l)
// RGBA8, rows packed. Laid out exactly as gpu_alloc_mipmaps does (level l at
// MAX(w_l, 8) * h_l * 4 bytes, rows padded to 8 texels) but filled here on
// the CPU from the caller's own images: no sceGxmTransferDownscale (async,
// and it streaked GX-uploaded textures) and no generated levels. w and h must
// be powers of two. Render-thread only.
int vglTexImage2DMipChain(GLuint tex_id, int w, int h, int levels, const void *const *rgba) {
	if (tex_id == 0 || tex_id >= TEXTURES_NUM || w <= 0 || h <= 0 || levels < 1 || !rgba)
		return -1;
	if ((w & (w - 1)) || (h & (h - 1)))
		return -2;
	size_t size = 0;
	for (int l = 0; l < levels; l++) {
		const int lw = MAX(w >> l, 1), lh = MAX(h >> l, 1);
		size += (size_t)MAX(lw, 8) * lh * 4;
	}
	uint8_t *data = (uint8_t *)gpu_alloc_mapped(size, VGL_MEM_MAIN);
	if (!data)
		return -3;
	uint8_t *dst = data;
	for (int l = 0; l < levels; l++) {
		const int lw = MAX(w >> l, 1), lh = MAX(h >> l, 1);
		const size_t dst_stride = (size_t)MAX(lw, 8) * 4, src_stride = (size_t)lw * 4;
		const uint8_t *src = (const uint8_t *)rgba[l];
		for (int y = 0; y < lh; y++)
			vgl_memcpy(dst + y * dst_stride, src + y * src_stride, src_stride);
		dst += dst_stride * lh;
	}
	texture *tex = &texture_slots[tex_id];
	if (tex->status == TEX_VALID)
		gpu_free_texture_data(tex);
	tex->type = GL_RGBA;
	tex->write_cb = NULL;
	tex->mip_count = levels;
	tex->use_mips = GL_TRUE;
	vglInitLinearTexture(&tex->gxm_tex, data, SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR, w, h, levels);
	tex->palette_data = NULL;
	tex->status = TEX_VALID;
	tex->data = data;
#ifndef TEXTURES_SPEEDHACK
	tex->last_frame = OBJ_NOT_USED;
#endif
	vglSetTexUMode(&tex->gxm_tex, tex->u_mode);
	vglSetTexVMode(&tex->gxm_tex, tex->v_mode);
	vglSetTexMinFilter(&tex->gxm_tex, tex->min_filter);
	vglSetTexMagFilter(&tex->gxm_tex, tex->mag_filter);
	vglSetTexMipFilter(&tex->gxm_tex, tex->mip_filter);
	vglSetTexLodBias(&tex->gxm_tex, tex->lod_bias);
	vglSetTexMipmapCount(&tex->gxm_tex, levels);
	return 0;
}

// Blended full-screen passes over the display (the blur and the knockout
// grey-out): fragment programs patched with src-alpha blending, the
// destination alpha kept.
static SceGxmShaderPatcherId efb_cfp_id[EFBCOPY_MAX_FP];
static SceGxmFragmentProgram *efb_cfprog[EFBCOPY_MAX_FP];
static int efb_ncfp = 0;

int vglEfbSetCompositePrograms(const SceGxmProgram *const *fps, int nfp) {
	if (!efb_vprog || !efb_vp_gxp || !fps || nfp <= 0 || nfp > EFBCOPY_MAX_FP)
		return -1;
	SceGxmBlendInfo blend;
	vgl_memset(&blend, 0, sizeof(blend));
	blend.colorMask = SCE_GXM_COLOR_MASK_ALL;
	blend.colorFunc = SCE_GXM_BLEND_FUNC_ADD;
	blend.alphaFunc = SCE_GXM_BLEND_FUNC_ADD;
	blend.colorSrc = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
	blend.colorDst = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	blend.alphaSrc = SCE_GXM_BLEND_FACTOR_ZERO;
	blend.alphaDst = SCE_GXM_BLEND_FACTOR_ONE;
	for (int i = 0; i < nfp; i++) {
		if (sceGxmShaderPatcherRegisterProgram(gxm_shader_patcher, fps[i], &efb_cfp_id[i]) < 0)
			return -2;
		if (sceGxmShaderPatcherCreateFragmentProgram(gxm_shader_patcher, efb_cfp_id[i], SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
				SCE_GXM_MULTISAMPLE_NONE, &blend, efb_vp_gxp, &efb_cfprog[i]) < 0)
			return -3;
	}
	efb_ncfp = nfp;
	return 0;
}

// One blended full-screen quad over the display: textures tex0 (unit 0) and
// tex1 (unit 1, 0 = none), p0 / p1 handed to the program in vUV23.zw. Drawn in
// a scene of its own on the same back buffer (so no vitaGL state is touched);
// the next draw resumes the display.
int vglEfbComposite(int fp_index, GLuint tex0, GLuint tex1, float p0, float p1) {
	if (!efb_vprog || fp_index < 0 || fp_index >= efb_ncfp || active_write_fb || system_app_mode)
		return -1;
	if (tex0 == 0 || tex0 >= TEXTURES_NUM || texture_slots[tex0].status != TEX_VALID)
		return -2;
	if (tex1 >= TEXTURES_NUM || (tex1 && texture_slots[tex1].status != TEX_VALID))
		return -2;
	if (needs_end_scene_get()) {
		vgl_efb_arm_display_depth_reload(NULL);
		vgl_efb_scene_end();
	}
	sceneReset();
	const float w = (float)DISPLAY_WIDTH, h = (float)DISPLAY_HEIGHT;
	sceGxmSetViewport(gxm_context, w * 0.5f, w * 0.5f, h * 0.5f, -h * 0.5f, 0.5f, 0.5f);
	sceGxmSetRegionClip(gxm_context, SCE_GXM_REGION_CLIP_OUTSIDE, 0, 0, DISPLAY_WIDTH - 1, DISPLAY_HEIGHT - 1);
	sceGxmSetCullMode(gxm_context, SCE_GXM_CULL_NONE);
	sceGxmSetFrontPolygonMode(gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
	sceGxmSetBackPolygonMode(gxm_context, SCE_GXM_POLYGON_MODE_TRIANGLE_FILL);
	sceGxmSetFrontDepthFunc(gxm_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetBackDepthFunc(gxm_context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(gxm_context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetBackDepthWriteEnable(gxm_context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetFrontStencilFunc(gxm_context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	sceGxmSetBackStencilFunc(gxm_context, SCE_GXM_STENCIL_FUNC_ALWAYS, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, 0, 0);
	SceGxmTexture t0 = texture_slots[tex0].gxm_tex;
	sceGxmTextureSetMinFilter(&t0, SCE_GXM_TEXTURE_FILTER_LINEAR);
	sceGxmTextureSetMagFilter(&t0, SCE_GXM_TEXTURE_FILTER_LINEAR);
	sceGxmTextureSetUAddrMode(&t0, SCE_GXM_TEXTURE_ADDR_CLAMP);
	sceGxmTextureSetVAddrMode(&t0, SCE_GXM_TEXTURE_ADDR_CLAMP);
	sceGxmSetVertexProgram(gxm_context, efb_vprog);
	sceGxmSetFragmentProgram(gxm_context, efb_cfprog[fp_index]);
	sceGxmSetFragmentTexture(gxm_context, 0, &t0);
	if (tex1) {
		SceGxmTexture t1 = texture_slots[tex1].gxm_tex;
		sceGxmTextureSetMinFilter(&t1, SCE_GXM_TEXTURE_FILTER_LINEAR);
		sceGxmTextureSetMagFilter(&t1, SCE_GXM_TEXTURE_FILTER_LINEAR);
		sceGxmTextureSetUAddrMode(&t1, SCE_GXM_TEXTURE_ADDR_CLAMP);
		sceGxmTextureSetVAddrMode(&t1, SCE_GXM_TEXTURE_ADDR_CLAMP);
		sceGxmSetFragmentTexture(gxm_context, 1, &t1);
	}
	float *vtx = (float *)gpu_alloc_mapped_temp(4 * 10 * sizeof(float));
	uint16_t *idx = (uint16_t *)gpu_alloc_mapped_temp(4 * sizeof(uint16_t));
	if (vtx && idx) {
		const float px[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
		const float py[4] = { 1.0f, -1.0f, 1.0f, -1.0f };
		const float pu[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
		const float pv[4] = { 0.0f, 1.0f, 0.0f, 1.0f };
		for (int i = 0; i < 4; i++) {
			float *o = vtx + i * 10;
			o[0] = px[i]; o[1] = py[i];
			o[2] = pu[i]; o[3] = pv[i];
			o[4] = pu[i]; o[5] = pv[i];
			o[6] = pu[i]; o[7] = pv[i];
			o[8] = p0; o[9] = p1;
			idx[i] = (uint16_t)i;
		}
		sceGxmSetVertexStream(gxm_context, 0, vtx);
		sceGxmDraw(gxm_context, SCE_GXM_PRIMITIVE_TRIANGLE_STRIP, SCE_GXM_INDEX_FORMAT_U16, idx, 4);
	}
	vgl_efb_scene_end();
	vgl_efb_request_restart();
	dirty_frag_unifs = GL_TRUE;
	dirty_vert_unifs = GL_TRUE;
	return 0;
}
