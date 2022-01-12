# Advanced DQM for a Drift Tube Detector

## LCP mod.A - Final Project

### Introduction
A novel approach for muons identification and track parameter estimation [[1]] consists in the implementation of an algorithm mixing artificial neural networks
and analytical methods on a FPGA. This algorithm is being tested on a
cosmic muon telescope at the Legnaro INFN National Laboratory (LNL), a detector composed by a
set of drift-tubes (DT) reproducing a small-scale replica of those in use at CMS. 

Recent developments in the search for new physics at LHC led to the implementation of a model-independent search strategy which exploits deep artificial neural networks [[2], [3], [4]]. The astonishing predictive power and flexibility of the New Physics Learning Machine (NPLM) algorithm can be conveyed to perform advanced Data Quality Monitoring (DQM) tasks [[8]].

### Resources
- Description of the experimental setup &rarr; [[5]] chapter 1 (1.2.1 explains the actual configuration) 
- How we track muons using scintillator signals &rarr; [[5]] section 2.2, 2.4 + chapter 3
- How the ML trigger algorithm reconstructs tracks (mean timer technique) &rarr; [[6]] (minimal summary) + [[7]] section 3.1 (complete documentation)
- Description of the NPLM algorithm &rarr; [[8]] chapter 2 (summary) + [[2]] (extended conceptual foundations)
- Application of NPLM to DQM &rarr; [[8]] chapter 3
### Outline

1. Build a 2D dataset with t<sub>drift</sub> and the crossing angle &theta; as features
   * using scintillator signals
   * using the ML algorithm reconstruction hidden within data
2. Study the correlation between the two features 
3. Test the performance of NPLM using the 2D dataset built in 1.
   * NN architecture?
   * What is the average training time for the algorithm?
   * If we put a constraint on the crossing angle, does the algorithm detect the anomaly in the drift time distribution or it correctly sees the correlation between the two features?
   * What if we cut the angular feature but keep all the time information? Does it see it as a discrepancy?

[1]: https://arxiv.org/abs/2105.04428
[2]: https://arxiv.org/abs/1806.02350
[3]: https://arxiv.org/abs/1912.12155
[4]: https://arxiv.org/abs/2111.13633
[5]: http://tesi.cab.unipd.it/65910/1/Franceschetto_Giacomo.pdf
[6]: https://github.com/spiccinelli/LCP_projects_Y3/blob/track_group6/Project.ipynb
[7]: http://cds.cern.ch/record/1073687/files/NOTE2007_034.pdf?version=1
[8]: https://nbviewer.org/github/niklai99/PredictiveLearning_applied_to_MuonChamberMonitoring/blob/master/THESIS/tesi.pdf

---

## Git Setup

1. Fork this repository clicking on the top-right button *Fork*.

2. Clone your forked repository &rarr; create a local repository in your machine.

   `git clone https://<YourToken>@github.com/<YourUsername>/LCP_modA_finalProject.git`

   where *YourUsername* it your GitHub username and *YourToken* is the token as copied from the GitHub webpage.

3. Get into the new folder:

   `cd LCP_modA_finalProject/`

4. Configure your username and email:

   `git config --global user.name "<YourUsername>"`

   `git config --global user.email "<YourEmail>"`

5. Define this repo as the upstream repository:

   `git remote add upstream https://<YourToken>@github.com/niklai99/LCP_modA_finalProject.git`

   Remember that in order to be able to push to the upstream you must be a contributor to this repo

6. Check

    `git remote -v`

7. Fetch for updates
  
   `git fetch upstream`

8. Check branches

    `git branch -vv`

## Git Development Cycle

1. Sync the main branch that will have the latest completed code:

   `git checkout main`

   `git fetch upstream`

   `git merge upstream/main`

2. Before starting to code in your machine make sure everything is up to date:

    `git pull`

3. ***Now you can start developing code***

4. Add files you want to commit (DO NOT add data files, weird folders and junk files):

    `git add <NewFile>`

5. Commit the tracked changes:

    `git commit -m "<MeaningfulMessage>"`

6. Push local changes into *your* remote repository on github (`origin`):

    `git push origin main`

7. Push local changes into the `upstream` repository:

    `git push upstream main`
