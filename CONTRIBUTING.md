# Contributing to PLImig (3D-PLI Mask and Inclination generation)

We would love your input to this repository! We want to make contributing to this project as easy and transparent as possible, whether it is:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

The development of PLImig is done on the Forschungszentrum Jülich GitLab.

Pull requests, issues, and feature requests are accepted via this GitLab. If you plan to contribute to PLImig, please follow the guidelines below.

## Seek support
The 3D-PLI Mask and Inclination generation is maintained by Jan André Reuter. For bug reports, feature requests, and pull requests, please read the instructions below. For further support, you can contact both per e-mail.

| Person           | E-mail address         |
| ---------------- | ---------------------- |
| Jan André Reuter | j.reuter@fz-juelich.de |


## Bug reports

We use GitLab issues to track public bugs. Report a bug by opening a new issue.
Write bug reports with detail, background, and add sample code if possible.

A good bug report should contain:

- A quick summary and/or background
- Steps to reproduce
    - Be specific!
    - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that did not work)

## Pull requests

Pull requests are the best way to propose changes to the codebase. When proposing a pull request, please follow these guidelines:

- Fork the repo and create your branch from master.
- If you've added code that should be tested, add tests.
- Ensure the test suite passes.
- Make sure your code lints.
- Issue that pull request!

## Testing

PLImig uses [googletest](https://github.com/google/googletest) to test the code for errors. In addition, [gcovr](https://gcovr.com/en/stable/) is used for linting.

Pull requests and commits to the master branch should be automatically tested using GitLab CI/CD with a simple workflow. If you want to test your code locally, follow the next steps:

1. Change your directory to the root of PLImig
2. If not done yet, follow the install instructions for further development of PLImig
3. Compile the project and run `make test`. Check if there are any errors left.

## Merging Pull Requests

This section describes how to cleanly merge your pull requests.

### 1. Rebase

From your project repository, merge from master and rebase your commits
(replace `pull-request-branch-name` as appropriate):

```
git fetch origin
git checkout -b pull-request-branch-name origin/pull-request-branch-name
git rebase master
```

### 2. Push

Update branch with the rebased history:

```
git push origin pull-request-branch-name --force
```

The following steps are intended for the maintainers:

### 3. Merge

```
git checkout master
git merge --no-ff pull-request-branch-name
```

### 4. Test

```
make test
```

### 5. Version
If necessary, update the version through the CMakeLists.txt:

```
git add CMakeLists.txt
git commit 
```

### 6. Push to master

git push origin master
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md) and [tqdm](https://github.com/tqdm/tqdm/blob/830cd7f9cb3e6fe9b1c3f601ff451debf9509916/CONTRIBUTING.md)
