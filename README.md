# Laboratory of Computational Physics - mod A

## Final Project

~ something about the project

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
