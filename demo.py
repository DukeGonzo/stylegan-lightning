import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import stylegan

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_model():
    gan = stylegan.GanTask(gamma=28, ppl_reg_every=4,
                           penalize_d_every=16, resolution=512)

    gan = gan.load_from_checkpoint(
        checkpoint_path="./lightning_logs/version_30/checkpoints/epoch=3-step=23157.ckpt",
        strict=False)

    return gan


@st.cache(allow_output_mutation=True)
def get_latents():
    with torch.no_grad():
        n = 256
        z = torch.randn([n, 512]).float()
        l = F.one_hot(torch.ones((n,)).long(), 2).float()
        latents_princesses = gan.mapping_net.forward(z, l)
        latent_mean = torch.mean(latents_princesses, 0)
        latent_plus_mean = latent_mean[None, None, :].repeat([1, 16, 1])
        fake_latents = latents_princesses[:, None, :].repeat([1, 16, 1])

        return latent_plus_mean, fake_latents


@st.cache
def load_pca_vectors():
    return np.load('./pca_vectors.npy')


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


gan = load_model()
latent_plus_mean, fake_latents = get_latents()
X_comp_norm = load_pca_vectors()


index = st.sidebar.slider('index', 0, 511, 0, 1)
shift = st.sidebar.slider('shift', -5.0, 5.0, 0.0, 0.5)
index_2 = st.sidebar.slider('index2', 0, 511, 0, 1)
shift_2 = st.sidebar.slider('shift2', -5.0, 5.0, 0.0, 0.5)
index_3 = st.sidebar.slider('index3', 0, 511, 0, 1)
shift_3 = st.sidebar.slider('shift3', -5.0, 5.0, 0.0, 0.5)
latent_index = st.sidebar.slider('latent_index', 0, len(fake_latents) - 1, 0, 1)
artefacts_shift = st.sidebar.slider('artefacts_shift', 0.0, 30.0, 0.0, 0.5)
cari_shift = st.sidebar.slider('cari_shift', 0.0, 12.0, 0.0, 0.5)
truncation = st.sidebar.slider('truncation', 0.0, 1.0, 0.7, 0.1)

latent = fake_latents[latent_index]
latent = latent_plus_mean[0].lerp(latent, truncation)

direction = X_comp_norm[index]
direction_2 = X_comp_norm[int(index_2)]
direction_3 = X_comp_norm[int(index_3)]

shifted_latent = latent + shift * direction + cari_shift * \
    X_comp_norm[int(0)] - artefacts_shift * X_comp_norm[int(11)]
shifted_latent_minus = latent - shift * direction + cari_shift * \
    X_comp_norm[int(0)] - artefacts_shift * X_comp_norm[int(11)]

shifted_latent += shift_2 * direction_2
shifted_latent_minus -= shift_2 * direction_2

shifted_latent += shift_3 * direction_3
shifted_latent_minus -= shift_3 * direction_3

latents_ = np.stack([shifted_latent_minus, latent, shifted_latent])

with torch.no_grad():
    noise = gan.synthesis_net.make_noise(3)
    rezults = gan.synthesis_net_ema.forward(
        torch.from_numpy(latents_).float(), noise)

out_minus = make_image(rezults[0])
src = make_image(rezults[1])
img = make_image(rezults[2])

st.title('Demo')

col1, col2, col3 = st.beta_columns(3)

col1.image(out_minus)
col2.image(src)
col3.image(img)
