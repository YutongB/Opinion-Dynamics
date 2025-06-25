# lmbh Opinion Dynamics

`lmbh` is an initialism for the opinion dynamics model introduced by the authors: Low, Melatos, Bu and Horstman.

## Installation with pixi

This repository depends on graph-tool.  Installation of a Python environment with graph-tool is a bit tricky, as it relies on some C++ dependencies. So we follow their [recommended instructions for installation by using pixi](https://graph-tool.skewed.de/installation.html#customizing-your-conda-environment), which makes it easier to use conda environments and conda-forge packages in a project-based environment (like ours).  At some point in the future, we may replace graph-tool with networkx to make installation and distribution simpler.

If you're on Windows, you must [use Windows Subsystem For Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). It is usually enough to run the command `wsl --install` from PowerShell, then rebooting. Then launch a WSL shell by searching for WSL in your start menu.  Now you can follow instructions as normal.

To install pixi:
```
curl -fsSL https://pixi.sh/install.sh | bash
```
Then source your shell with `source ~/.bash_profile` or `exec bash` or `exec zsh` depending on shell.  Use `exec bash` if you are unsure.

Clone this repo to some folder, for example `lmbh`: `git clone https://github.com/YutongB/Opinion-Dynamics lmbh`

In the repo folder, run `pixi install`. That should install everything for this project.

Test that installation works by opening `demo.ipynb`.  You can use `pixi shell`, then `jupyter notebook` to open the notebook.  Exit the pixi shell with `exit`.  Or, use [Visual Studio Code's Jupyter extension](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) by running `code .`.  In vscode's Jupyter interface, select the "default" conda environment created by pixi.  It will take ~15 seconds to load the `lmbh` library for the first time, as importing the graph-tool library for the first time requires some installation to happen behind the scenes.

## Web App
We made an web application for this simulation, it does not do everything that the code is able to do, but it can run simple simulation and basic analysis. 
Simulation Editor web application: https://opinion-workshop.vercel.app

## Papers
1. "Discerning media bias within a network of political allies and opponents: The idealized example of a biased coin", Nicholas Kah Yean Low, Andrew Melatos. [Physica A](https://www.sciencedirect.com/science/article/abs/pii/S037843712100933X) [arXiv](https://arxiv.org/abs/2112.10160)

2. "Vacillating about media bias: changing one's mind intermittently within a network of political allies and opponents", Nicholas Kah Yean Low, Andrew Melatos. [Physica A](https://www.sciencedirect.com/science/article/abs/pii/S0378437122005404) [arXiv](https://arxiv.org/abs/2207.00372)

3. "Discerning media bias within a network of political allies and opponents: Disruption by partisans", Yutong Bu, Andrew Melatos. [Physica A](https://www.sciencedirect.com/science/article/abs/pii/S0378437123005137) [arXiv](https://arxiv.org/abs/2307.16359)

4. "Discerning media bias within a network of political allies: an analytic condition for disruption by partisans", Jarra Horstman, Andrew Melatos, Farhad Farokhi. [Physica A](https://www.sciencedirect.com/science/article/abs/pii/S0378437125003310) [arXiv](https://arxiv.org/abs/2505.06959)

5. "How trust networks shape students' opinions about the proficiency of artificially intelligent assistants", Yutong Bu, Andrew Melatos, Robin Evans. [arXiv](https://arxiv.org/abs/2506.19655)

## Contact 
If you have any questions about our work, setting up the code, or wish to reach out to the group, please contact Yutong Bu at buy1@student.unimelb.edu.au
