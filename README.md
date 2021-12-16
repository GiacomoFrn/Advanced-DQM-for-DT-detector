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

   `git remote add upstream https://github.com/niklai99/LCP_modA_finalProject.git`

6. Check

    `git remote -v`

7. Fetch for updates
  
   `git fetch upstream`

8. Check branches

    `git branch -vv`

I have created a new branch "dataset_script" in which I will upload the script I am refining. Once it will be fully
operational I will merge the branch with the main branch.

Whenever you will be developing code, please create a different branch (with a meaningful name) and request for the
merging with main once you have completed it.

I'm not sure whether you are enabled to create branches in the upstream or only I have the permession to do so. In the
latter case, just create a new branch in your own fork and ask me to create a new branch in the upstream whenever you
want to submit your code to the others.

## Git Development Cycle

1. Sync the main branch that will have the latest completed code:

   `git checkout main`

   `git fetch upstream`

   `git merge upstream/main`

2. Sync the branch you are working in:

   `git checkout <BranchName>`

   `git fetch upstream <BranchName>`

   `git merge upstream/<BranchName>`

3. Before starting to code in your machine make sure everything is up to date:

    `git pull`

4. If you need some code that has been moved to the main then merge the main branch in your working branch:

   `git merge main`

5. ***Now you can start developing code***

6. Add files you want to commit (DO NOT add data files, weird folders and junk files):

    `git add <NewFile>`

7. Commit the tracked changes:

    `git commit -m "<MeaningfulMessage>"`

8. Push local changes into *your* remote repository on github (`origin`):

    `git push origin <BranchName>`

9. Once you finished a specific task, or reached an important point where you want the other members to have your code,
   then propagate your changes to this repository (`upstream`) via pull request on GitHub.