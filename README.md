# Opinion Dynamics 

## Installation with pixi

graph-tool makes installation a little tricky. Luckily, pixi _should_ make things easy for us.

If you're on Windows, use WSL. https://graph-tool.skewed.de/installation.html#windows

Install pixi, which uses conda environments and conda-forge packages, but is project-centric, which works well for us

```
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bash_profile
```
or `exec zsh` or whatever

In this folder, run `pixi install`. That should install everything. Test that it works by opening demo.ipynb in this folder, vscode should detect the current environment (or you can use `pixi shell` to activate the environment for the current project, and `exit` to exit it)

## Installation
Ensure you have anaconda correctly installed.
1. Create the environment from the `environment.yml` file.
```
conda env create -f environment.yml
```
2. Activate the new environment `conda activate opinion-dynamics`
3. Run `simulate.py -help` to get started.

### Manual installation
We use the packages `scipy numpy graph_tool matplotlib pandas jupyter`


## Papers
Model : https://arxiv.org/abs/2112.10160
Intermittency : https://arxiv.org/abs/2207.00372
Disruption by partisans : https://arxiv.org/abs/2307.16359
Analytic condition : https://arxiv.org/abs/2505.06959

## Contact 
If you have any questions about our work, setting up the code, or wish to reach out, please contact me at : buy1@student.unimelb.edu.au





