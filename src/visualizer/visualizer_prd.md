# Visualizer PRD

Below is a practical, end-to-end recipe that teams have found useful for explaining a ChuChu-Rocket puzzle to themselves by literally “lighting up” the parts of the network that matter. Everything is framed for the exact architecture in our PRDs, so you can copy-paste most of it.

1 · Instrument the model
Layer	What to capture	Why
CNN Encoder (4 × Conv-BN-ReLU)	Output feature map of each conv block	Lets you see which board tiles trigger which filters. ai_overview
Cat Set Encoder	Per-cat 32-d embedding before max-pool and the pooled vector	Reveals which cat is the “max-winner” on each latent dim. ai_overview
Fusion MLP	128-d fused embedding	Good substrate for dimensionality-reduction plots. state_fusion_prd
Policy Head (pre-softmax)	700 × 1 logits	Can be decoded back to a board-heat-map of arrow “desire”. policy_prd


python
Copy
Edit
# register_forward_hooks.py
ACTIVATIONS = {}
def save_tensor(name):
    def _hook(_, __, out):
        ACTIVATIONS[name] = out.detach().cpu()
    return _hook

model.cnn_encoder[3].register_forward_hook(save_tensor("cnn_block1"))
model.cnn_encoder[7].register_forward_hook(save_tensor("cnn_block2"))
model.cat_set_encoder.register_forward_hook(save_tensor("cat_embeddings_raw"))
model.fusion_mlp.register_forward_hook(save_tensor("fused_latent"))
model.policy_head.register_forward_hook(save_tensor("policy_logits"))
2 · Run a specific puzzle through the net
python
Copy
Edit
with torch.no_grad():
    _ = model(perception_output_example)  # ACTIVATIONS now filled
(perception_output_example is the 28 -chan tensor for your puzzle; comes out of GameStateProcessor.) perception_prd

3 · Visualise what you caught
3.1 Heat-map the CNN feature maps on the board
Pick one conv block’s activation: A = ACTIVATIONS["cnn_block2"] → shape [C, H, W].

For each channel c, upsample to board size (nearest-neighbour) and normalise 0-1.

Overlay it on the board sprite image with plt.imshow(alpha=0.6).

Tip: The channels often group into “wall detectors”, “horizontal-flow detectors”, “cat-danger detectors”, etc. Scrolling through them quickly shows why the policy avoids certain placements.

3.2 Show which cat dominated the max-pool
Take the per-cat embeddings [N_cats, 32] captured before pooling. For each latent dim find torch.argmax across the cat axis → that tells you which cat contributed the max. Display a 32-element bar chart whose colour encodes the winning cat ID.

3.3 Project the fused 128-d latent
Collect latent vectors from many puzzles (or many ticks of one puzzle) and run PCA/t-SNE → scatter plot, coloured by “puzzle solved / unsolved” or “remaining mice”. Clusters often align with high-level game phases.

3.4 Turn raw policy logits into a board overlay
Reshape the 700-dim logits back into [5 actions × H × W] using the indexing scheme in the Policy PRD. policy_prd

Softmax over the 5 directions per tile to see preferred direction ignoring erase.

For each tile draw an arrow whose thickness ∝ that tile’s max direction prob.

(Drawing arrows makes behavioural intent obvious: the map literally shows the routing field the agent wants to build.)

3.5 Saliency / attribution for a single decision
Feed the same state back through Captum (Integrated Gradients or Gradient × Input) on the grid tensor input. Map the resulting attribution back onto the 28-channel board; you’ll get a red-hot outline around walls or cats that “mattered” for that arrow choice.

4 · Recommended toolchain
Need	Off-the-shelf helper
Network graph diagram	Netron – drop the *.pth to get an interactive JSON/PNG export.
Activation capture	PyTorch forward hooks (built-in).
Heat-maps & arrows	matplotlib (imshow, quiver).
Embedding projection	sklearn.manifold.TSNE or PCA.
Attribution	Captum (IntegratedGradients, GradientShap).
Interactive dashboards	TensorBoard (add_embedding, add_image).

(Everything above is GPU-cheap and keeps you inside the PyTorch + matplotlib stack, so it works in the same notebook you already use for training.)

5 · Putting it together in practice
Write a one-liner wrapper visualise_state(game_state) that

builds the perception tensors,

runs the model with hooks,

dispatches to the four visualisers above.

Drop it at key milestones (after BC-Set, after PPO 1 M steps) to watch how understanding evolves.

Use it as a debugging lens: if a puzzle fails, visualising immediately tells you whether the CNN missed a wall, the cat encoder mis-ranked a threat, or the policy head just chose a low-prob action.

Once you’ve done this a couple of times, you’ll find the network’s behaviour becomes intuitive—the activations essentially act as high-level annotations on the board. Enjoy the light-show!