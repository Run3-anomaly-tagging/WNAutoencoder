import matplotlib.pyplot as plt
import os

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def plot_losses(training_losses, validation_losses, png_path):
    ensure_dir(os.path.dirname(png_path))
    epochs = list(range(len(training_losses)))
    plt.figure()
    plt.plot(epochs, training_losses,  label="Training",   linewidth=2)
    plt.plot(epochs, validation_losses, label="Validation", linestyle="--", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(png_path,dpi=200)
    plt.close()

def plot_epoch_1d(data, mcmc, outdir, epoch, features, bins):
    ensure_dir(outdir)
    for feat in features:
        plt.hist(data[:, feat], bins=bins, histtype='step', density=True, label='data')
        plt.hist(mcmc[:, feat], bins=bins, histtype='step', density=True, label='MCMC')
        plt.legend()
        plt.title(f'Epoch {epoch} — feat {feat}')
        plt.xlim(-4.,4.)
        plt.ylim(0.,1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'epoch{epoch}_feat{feat}.png'),dpi=200)
        plt.savefig(os.path.join(outdir, f'epoch{epoch}_feat{feat}.pdf'))
        plt.close()

def plot_epoch_2d(data, mcmc, outdir, epoch, pairs):
    ensure_dir(outdir)
    for pair in pairs:
        x = pair[0]
        y = pair[1]
        plt.scatter(data[:, x], data[:, y], alpha=.2, label='Data')
        plt.scatter(mcmc[:, x], mcmc[:, y], alpha=.2, label='MCMC')
        plt.xlim(-4.,4.)
        plt.ylim(-4.,4.)
        plt.legend()
        plt.title(f'Epoch {epoch} — (feat {x}, feat {y})')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'epoch{epoch}_2d_{x}_{y}.png'),dpi=200)
        plt.savefig(os.path.join(outdir, f'epoch{epoch}_2d_{x}_{y}.pdf'))
        plt.close()

