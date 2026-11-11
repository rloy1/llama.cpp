#include "models.h"

#include "llama-memory-recurrent.h"

// Hybrid: LLM_ARCH_QWEN35's gated-delta-net linear-attention layers,
// unchanged, interleaved with DeepSeek-style MLA full-attention layers in
// place of LLM_ARCH_QWEN35's own plain gated-GQA full attention. See
// models.h's declaration and /mnt/models/projects/transmla/QWEN35_MLA_GUIDE.md
// for why this needs to be its own architecture rather than a flag on
// LLM_ARCH_QWEN35 or LLM_ARCH_DEEPSEEK2.
//
// build_layer_attn_linear/build_layer_ffn/build_norm_gated/build_qkvz below
// are verbatim copies of qwen35.cpp's own methods -- the 24 unconverted
// layers use identical math, this file just needs its own copies since
// C++ has no cross-file private-method sharing between sibling classes.
llm_build_qwen35_mla::llm_build_qwen35_mla(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    // Every non-recurrent layer here is MLA (is_mla() is unconditionally
    // true for this arch -- see the hparams block in llama-model.cpp),
    // unlike DeepSeek2/Kimi-Linear which support both. No is_mla() branch
    // needed, just always use the k-only hybrid memory input.
    auto * inp = build_inp_mem_hybrid_k();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn_mla(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_qwen35_mla::build_layer_attn_mla(
        llm_graph_input_attn_k * inp,
        ggml_tensor *            cur,
        ggml_tensor *            inp_pos,
        int *                    sections,
        int                      il) {
    const int64_t n_embd_head_v  = hparams.n_embd_head_v_mla(); // real per-head width (256), also gate's per-head width
    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    // The rope-carrying slice is n_rope_ranks blocks of the model's true
    // rotary width, where n_rope_ranks is how many cross-head principal
    // components per rope frequency reach the exact path. Both numbers are
    // already in the gguf, so this needs no new hparam:
    //
    //   rope width = key_length_mla - value_length_mla
    //   tiles      = rope width / n_rot
    //
    // At n_rope_ranks=1 that is 320-256 = 64 = n_rot, one tile, i.e. exactly
    // the previous behaviour -- existing models are unaffected. At 4 ranks it
    // is 512-256 = 256 with n_rot still 64, so four tiles.
    //
    // Why it matters: stage 1 of the conversion costs 0.0144 relative
    // attention error at one rank and exactly 0.0000 at four (measured,
    // stage_decompose.py). The rank ladder bottoms out at 11-12% unsupported
    // claims against base's 8% at every rank from 768 up, because that floor is
    // the one-rank positional loss and no amount of kv_lora_rank touches it.
    const int64_t n_embd_head_qk_rope = n_embd_head_k_mla - n_embd_head_v;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t n_rot_true          = hparams.n_rot();
    GGML_ASSERT(n_embd_head_qk_rope % n_rot_true == 0);
    const int64_t n_rope_tiles        = n_embd_head_qk_rope / n_rot_true;
    // PER-LAYER latent rank: a ragged allocation gives each converted layer its
    // own kv_lora_rank, so this is indexed by il rather than read as the
    // model-wide scalar (which is now the maximum across layers). Getting this
    // wrong would not fail loudly -- it would silently build views of the wrong
    // width over the cache -- so it must track the tensor shapes in
    // llama-model.cpp, which use hparams.n_lora_kv_l(i) for the same reason.
    const int64_t kv_lora_rank        = hparams.n_lora_kv_l(il);
    const int64_t n_head_kv_real      = hparams.n_head_kv_real;
    const int64_t latent_dim          = n_head_kv_real * n_embd_head_v;

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k_mla));

    // --- gate: extracted from the raw doubled wq, per-head-interleaved
    // layout [head0_query(n_embd_head_v), head0_gate(n_embd_head_v),
    // head1_query, head1_gate, ...] -- identical layout and extraction to
    // qwen35.cpp's build_layer_attn, since it comes from the same
    // TransMLA-untouched q_proj_original. The "query" half of wq's output
    // is used below to compute q_norm's exact RMS denominator (see
    // qwen35_mla.py's module docstring: the elementwise weight-scale half
    // of q_norm is already folded into wq_b's exported weights, but the
    // per-token magnitude correction was never carried into this graph --
    // it's a per-head scalar and commutes through everything downstream,
    // same as the docstring says, so it's applied once right here).
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head_v, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head_v * 2,
        ggml_element_size(Qcur_full) * n_embd_head_v * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head_v);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head_v * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    ggml_tensor * q_raw = ggml_view_3d(ctx0, Qcur_full, n_embd_head_v, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head_v * 2,
        ggml_element_size(Qcur_full) * n_embd_head_v * 2 * n_head,
        0);
    q_raw = ggml_cont(ctx0, q_raw);
    const float eps_norm = hparams.f_norm_rms_eps;
    ggml_tensor * q_rms_denom = ggml_sqrt(ctx0, ggml_scale_bias(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, q_raw)), 1.0f, eps_norm));
    cb(q_rms_denom, "q_rms_denom", il);

    // --- compressed, rope-absorbed query, ready for the MLA nope/rope split
    ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_b, cur);
    cb(Qcur, "Qcur_mla", il);

    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
    Qcur = ggml_div(ctx0, Qcur, q_rms_denom);
    Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head_k_mla * n_head, n_tokens);
    cb(Qcur, "Qcur_q_normed", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
        ggml_row_size(Qcur->type, n_embd_head_k_mla),
        ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
    cb(q_nope, "q_nope", il);

    ggml_tensor * q_pe = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
        ggml_row_size(Qcur->type, n_embd_head_k_mla),
        ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
        ggml_row_size(Qcur->type, n_embd_head_qk_nope));
    cb(q_pe, "q_pe", il);

    // --- exact K-rope AND K-nope paths: raw k_proj -> real per-head
    // RMSNorm -> the precomputed rotation matrix that reproduces
    // Qwen35PartialRope._freqfold_rotate exactly (see qwen35_mla.py's
    // k_rotation_rope_export / k_rotation_nope_export docstrings for the
    // identity-probe derivation that makes this a single matmul instead
    // of needing the PCA rotation's reshape/permute dance replicated in
    // ggml). Normalize once here, then rotate into both the rope slice
    // and the nope slice that feeds kv_cmpr's K-contribution below --
    // the rotation mixes across real-kv-heads, so normalizing first (not
    // trying to correct after) is what makes this exact rather than
    // approximate, same reasoning as Q's fix above, just needing this
    // extra rotation step since Q never gets mixed across anything
    // downstream and K does.
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].wk, cur);
    cb(k_raw, "k_raw", il);
    k_raw = ggml_reshape_3d(ctx0, k_raw, n_embd_head_v, n_head_kv_real, n_tokens);
    ggml_tensor * k_normed = build_norm(k_raw, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(k_normed, "k_normed", il);
    k_normed = ggml_reshape_2d(ctx0, ggml_cont(ctx0, k_normed), latent_dim, n_tokens);

    ggml_tensor * k_pe = ggml_mul_mat(ctx0, model.layers[il].wk_rope_rot, k_normed);
    cb(k_pe, "k_pe_exact", il);
    k_pe = ggml_reshape_3d(ctx0, k_pe, n_embd_head_qk_rope, 1, n_tokens);

    // --- KV compression: K-nope's and V's separate, exact contributions
    // to kv_cmpr, summed. Replaces a single fused hidden_state->
    // kv_lora_rank matrix, which had no place left to inject K's RMS
    // correction once K-nope and V were fused together (see
    // qwen35_mla.py's module docstring). K's contribution takes the
    // already-exact normed+rotated k_nope_exact as input; V's takes the
    // raw hidden state directly, same as the old fused matrix did, since
    // V never had a norm to correct in the first place.
    // Full fp32 precision explicitly requested on all three of these:
    // kv_cmpr_k and kv_cmpr_v are individually-computed halves of a value
    // that used to come from one single fused matmul, and get summed back
    // together right after -- if the two halves are individually larger
    // than their sum (real cancellation, expected given they used to
    // share one compressed representation), any per-operand precision
    // loss gets amplified by that cancellation. Cheap insurance against
    // exactly that.
    ggml_tensor * k_nope_exact = ggml_mul_mat(ctx0, model.layers[il].wk_nope_rot, k_normed);
    ggml_mul_mat_set_prec(k_nope_exact, GGML_PREC_F32);
    cb(k_nope_exact, "k_nope_exact", il);
    ggml_tensor * kv_cmpr_k = ggml_mul_mat(ctx0, model.layers[il].wkv_a_from_k_normed, k_nope_exact);
    ggml_mul_mat_set_prec(kv_cmpr_k, GGML_PREC_F32);
    cb(kv_cmpr_k, "kv_cmpr_k", il);
    ggml_tensor * kv_cmpr_v = build_lora_mm(model.layers[il].wkv_a_v, cur);
    ggml_mul_mat_set_prec(kv_cmpr_v, GGML_PREC_F32);
    cb(kv_cmpr_v, "kv_cmpr_v", il);
    ggml_tensor * kv_cmpr = ggml_add(ctx0, kv_cmpr_k, kv_cmpr_v);
    cb(kv_cmpr, "kv_cmpr_exact", il);

    // --- MRoPE (interleaved), same call qwen35.cpp's own full-attention
    // layers use -- Qwen3.5's own partial_rotary_factor already limits
    // rope to exactly n_embd_head_qk_rope dims, so q_pe/k_pe (already that
    // width) are roped in full here, unlike qwen35.cpp's own call which
    // ropes only a partial slice of a wider tensor.
    // Fold the tile axis into the head axis so one call ropes every rank
    // block identically. ggml_reshape needs a contiguous source and q_pe is a
    // strided view into Qcur, hence the cont.
    if (n_rope_tiles > 1) {
        q_pe = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q_pe),
                               n_rot_true, n_rope_tiles * n_head, n_tokens);
        k_pe = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k_pe),
                               n_rot_true, n_rope_tiles, n_tokens);
    }
    q_pe = ggml_rope_multi(
            ctx0, q_pe, inp_pos, nullptr,
            n_rot_true, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    k_pe = ggml_rope_multi(
            ctx0, k_pe, inp_pos, nullptr,
            n_rot_true, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    if (n_rope_tiles > 1) {
        q_pe = ggml_reshape_3d(ctx0, q_pe, n_embd_head_qk_rope, n_head, n_tokens);
        k_pe = ggml_reshape_3d(ctx0, k_pe, n_embd_head_qk_rope, 1, n_tokens);
    }
    cb(q_pe, "q_pe_roped", il);
    cb(k_pe, "k_pe_roped", il);

    // --- MLA absorption: fold q_nope through wk_b so attention happens
    // directly in kv_lora_rank space, without ever materializing per-head
    // K -- this is what keeps the KV cache genuinely compressed.
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    cb(q_nope, "q_nope_perm", il);

    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
    cb(q_nope_absorbed, "q_nope_absorbed", il);

    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

    ggml_tensor * Qcur_final = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Qcur_final, "Qcur_final", il);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr_reshape", il);

    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = kv_cmpr;
    cb(Vcur, "Vcur", il);

    // wo deliberately NOT passed to build_attn -- the sigmoid gate has to
    // be applied to the raw attention output before o_proj (matches
    // Qwen3.5's own gated-attention layers in qwen35.cpp), so o_proj
    // happens manually below instead of inside build_attn. build_attn's
    // wo parameter is optional specifically for this reason (verified in
    // llama-graph.cpp's llm_graph_input_attn_k overload: `if (wo) { ... }`).
    cur = build_attn(inp, nullptr, nullptr,
                Qcur_final, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35_mla::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen35_mla::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35_mla::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels, n_seqs, conv_states_all->nb[1],
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen35_mla::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}
