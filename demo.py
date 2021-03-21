import argparse
import sys
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import stylegan

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_model(checkpoint_path: str):
    gan = stylegan.GanTask(gamma=28, ppl_reg_every=4,
                           penalize_d_every=16, resolution=512)

    gan = gan.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False)

    return gan.eval()

@st.cache()
def load_pca_vectors():
    return np.load('./pca_vectors.npy')


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Data processing pipeline')
    parser.add_argument('--model', help='Path to model checkpoint', required=True, type=str)

    return parser.parse_args(args)


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(1, 2, 0)
        .to("cpu")
        .numpy())


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]

    args = parse_args(args)

    gan = load_model(args.model)

    @st.cache(allow_output_mutation=True)
    def get_latents():
        with torch.no_grad():
            n = 128
            z = torch.randn([n, 512]).float()
            l = F.one_hot(torch.ones((n,)).long(), 2).float()
            latents_princesses = gan.mapping_net.forward(z, l).detach()
            latent_mean = torch.mean(latents_princesses, 0)
            latent_plus_mean = latent_mean[None, None, :].repeat([1, 16, 1])
            fake_latents = latents_princesses[:, None, :].repeat([1, 16, 1])

            return latent_plus_mean, fake_latents

    latent_plus_mean, fake_latents = get_latents()
    PCA_vectors = load_pca_vectors()

    indexes_count = 5
    shifts = []

    latent_index = st.sidebar.slider('Face', 0, len(fake_latents) - 1, 0, 1)
    truncation = st.sidebar.slider('Truncation', 0.0, 1.0, 0.7, 0.1)

    latent = fake_latents[latent_index]
    latent = latent_plus_mean[0].lerp(latent, truncation)

    shifted_latent = latent
    shifted_latent_minus = latent

    for index in range(indexes_count):
        st.sidebar.subheader(f'Direction {index}')
        sb_col1, sb_col2, sb_col3 = st.sidebar.beta_columns((1,3,3))
        is_index_enabled = sb_col1.checkbox('E', True, key=f'enabled_{index}')
        is_index_fixed = sb_col1.checkbox('F', False, key=f'fixed_{index}')
        PCA_index = sb_col2.slider('Index', 0, 511, 0, 1, key=f'direction_{index}')
        shift = sb_col3.slider('Shift', -5.0, 5.0, 0.0, 0.5, key=f'shift_{index}')

        if is_index_enabled:
            PCA_direction = PCA_vectors[PCA_index]
            shifted_latent += PCA_direction * shift
            
            if is_index_fixed:
                shifted_latent_minus += PCA_direction * shift
            else:
                shifted_latent_minus -= PCA_direction * shift

    st.sidebar.text('E - Enable direction')
    st.sidebar.text('F - Shift in the same direction for both images')

    latents_ = np.stack([shifted_latent_minus, latent, shifted_latent])

    with torch.no_grad():
        noise = gan.synthesis_net.make_noise(3)
        results = gan.synthesis_net_ema.forward(
            torch.from_numpy(latents_).float(), noise)

    out_minus = make_image(results[0])
    src = make_image(results[1])
    img = make_image(results[2])

    st.title('Demo')

    col1, col2, col3 = st.beta_columns(3)

    col1.image(out_minus, caption="Shifted in the negative direction")
    col2.image(src, caption="Original")
    col3.image(img, caption="Shifted in the positive direction")

if __name__ == "__main__":
    # execute only if run as a script
    main()
