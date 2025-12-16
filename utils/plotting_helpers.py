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

def plot_aux_scatter(data, mcmc, savedir, epoch, aux_dim=2):
    """
    Plot 2D scatter of the last two auxiliary features (i.e., indices -2 and -1).
    aux_dim indicates how many auxiliary dimensions are at the end of the vector.
    """
    # Require at least two auxiliary features and enough total features
    if aux_dim < 2 or data.shape[1] < aux_dim or mcmc.shape[1] < aux_dim:
        print(f"Warning: Need at least aux_dim>=2 and arrays with >= aux_dim features (got aux_dim={aux_dim})")
        return

    ensure_dir(savedir)

    x_data, y_data = data[:, -2], data[:, -1]
    x_mcmc, y_mcmc = mcmc[:, -2], mcmc[:, -1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x_data, y_data, alpha=0.3, s=10, label='Data', rasterized=True)
    ax.scatter(x_mcmc, y_mcmc, alpha=0.3, s=10, label='MCMC', rasterized=True)

    ax.set_xlabel(f'Auxiliary feature {aux_dim-1} (Gaussianized)', fontsize=12)
    ax.set_ylabel(f'Auxiliary feature {aux_dim} (Gaussianized)', fontsize=12)
    ax.set_title(f'Auxiliary Features Scatter (last two) - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(savedir, f"aux_scatter_epoch{epoch}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

