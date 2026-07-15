# TinyTRACE Architecture Contract

This file records the executable interface for Phase 1. It complements, but
does not replace, `trace_lightwieght.md`.

## Visual path

1. Input frames: `[B, T, 3, H, W]`, floating point RGB in `[0, 1]`.
2. MobileCLIP-S0 preprocessing: bilinear resize to `256 x 256`. The official
   MobileCLIP v1 transform applies resize, center crop, and tensor conversion;
   it does not apply an additional normalization transform.
3. Frozen MobileCLIP-S0 spatial tower: use `forward_embeddings`,
   `forward_tokens`, and `conv_exp`, before the global-pooling head.
4. Expected S0 spatial output: `[B*T, 1024, 8, 8]`.
5. Flattened patch features: `[B*T, 64, 1024]`.
6. Learned slot compression: `[B*T, 64, 1024] -> [B*T, 4, d_model]`.

Calling MobileCLIP's public `encode_image()` is not valid for this path because
it returns one globally pooled embedding and removes the spatial token axis.

## Decoder prefix

Each scalar frame timestamp is serialized as fixed-width `0000.0`, producing
six IDs from the shared 13-token time vocabulary. `<sync>` is not included in
frame-time metadata because it is reserved for output head switching.

For each frame, tokens are concatenated in this order:

`[4 compressed visual slots, 6 discrete frame-time embeddings]`

The frame groups are flattened in chronological order, then instruction and
teacher-forced/generated event tokens are appended:

`[frame 1 visual+time, ..., frame T visual+time] + instruction + events`

## Frozen-module invariant

All MobileCLIP parameters have `requires_grad=False`, and its BatchNorm layers
remain in evaluation mode even when the parent TinyTRACE model is training.
