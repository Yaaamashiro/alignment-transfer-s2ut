# GitHub access from Colab

The notebooks fetch their configured `REPO_REVISION` over HTTPS and check out that revision in detached HEAD mode. A clean runtime clones with `--no-checkout`. An existing clone fetches again before checkout.

## Configure Colab Secrets

1. In GitHub, open **Settings → Developer settings → Personal access tokens → Fine-grained tokens** and generate a token. Select the repository owner and grant access to `alignment-transfer-s2ut` under **Only select repositories**. Set **Contents** to **Read-only** and choose an expiration date. [GitHub token guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
2. In Colab, open **Secrets** using the key icon. Create `GITHUB_TOKEN`, paste the token into its value, and enable **Notebook access** for each notebook you use.
3. Run setup and check that `clone-alignment` on a fresh runtime, `fetch-alignment`, and `checkout-alignment` complete. For a private repository, check the secret's notebook access, token expiration, selected repository, and any required organization approval if the Git command fails.

The Git subprocess receives a temporary `http.extraHeader` scoped to the repository's HTTPS URL through `GIT_CONFIG_COUNT`. The worker removes token values from Git output before it reaches the notebook and log files. Public repositories can use anonymous access when the secret is unavailable. The benchmark and fairseq use their public sources by default. If the benchmark later requires authentication, add its exact HTTPS URL to `GITHUB_AUTH_REPOS` in the notebook settings and grant the token read access to that repository. [Git configuration reference](https://git-scm.com/docs/git-config)

## Before changing visibility

- Save the updated `.ipynb` files you use in your own Drive or download a local copy. Open each copy in Colab and enable its secret access.
- Run the setup cells on a fresh runtime and on an existing checkout. Verify that Git fetches and checks out the expected `REPO_REVISION`. Inspect the bootstrap log and the stored `origin` URL for credential-free output.
- A successful read of the public repository confirms that the setup works while it is public. To verify private access separately, use a private repository that you already control and a token with read access to it. After loading the notebook's logging and path cells, set `REPO_URL` to that repository's HTTPS URL and call `execute('git', 'ls-remote', REPO_URL, 'HEAD')`. Restore the original `REPO_URL` before continuing with setup.

## Open and resume after the visibility change

Download the `.ipynb` while signed in to GitHub and use **File → Upload notebook** in Colab, or open the copy saved in Drive. Enable access to `GITHUB_TOKEN` for the open notebook. [Colab FAQ](https://research.google.com/colaboratory/faq.html#using-colab)

Select the original experiment ID, settings, Drive paths, and source revision, then rerun from the setup cells. `training.ipynb` uses the saved Drive snapshots to resume and `evaluate.ipynb` restores the saved checkpoints for evaluation. Use the original revision for an in-progress run.
