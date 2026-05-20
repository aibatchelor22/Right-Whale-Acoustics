import matplotlib.pyplot as plt
import librosa.display


def plot_spectrogram(
    spec,
    sr,
    hop_length,
    fmin,
    fmax,
    title=None
):

    plt.figure(figsize=(8, 5))

    img = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        fmin=fmin,
        fmax=fmax,
        cmap="winter"
    )

    plt.colorbar(
        img,
        format="%+2.0f dB"
    )

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    plt.show()
