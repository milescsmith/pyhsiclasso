# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.13.0] - 2024-08-23

### Added

- More type hinting

### Changed

- Moved from `poetry` to `pdm` for packaging
- Removed a lot of redundant/unused files (e.g. `.flake8`)
- Replaced use of `flake8`/`black`/`isort` with `ruff`
- Changed a lot of variable names to conform to ruff's suggested styling (i.e. `X_in` to `x_in`)
- All instances of `numpy.typing.ArrayLike` to the more appropriate `numpy.typing.NDArray`
- Added the newest incarnation of the logging submodule

### Removed

- example folder
- `pyhsiclasso.api.HSICLasso._check_args`

## [1.12.0] - 2024-02-09

### Added
- `dump_dict` method to the `HSICLasso` class, which outputs the same data as `dump` did, but as a dict

### Changed
- Refactor `dump` to use a `rich.table.Table` to display the data instead of using (broken) custom formatting


## [1.11.0] - 2024-02-08

### Changed
- Any instance of `output_list` have been changed to `output` (still need a better name...)

### Fixed
- More fixes for loading data
- Revised unit tests to account for new input methods


## [1.10.0] - 2024-02-07

### Added
- Use `loguru` for logging
    - add submodule to handle log formatting
- More type hints

### Changed
- Essentially rewrote data input
- Removed `icecream`


## [1.9.0] - 2024-02-06

### Changed
- Finished replacing `unittest` with `pytest`


## [1.8.0] - 2024-02-05

### Added
- More saveguards to data import

### Changed
- Started replacing some tests written for `unittest` with equivalents in `pytest`


## [1.7.0] - ?
?


## [1.6.0] - ?
?


## [1.5.0] - 2021-10-11

### Added
- Ability to directly import data into a `HSICLasso` object from a pandas `DataFrame`
- Some typing information to functions
- Black formatting

### Changed
- Use Poetry instead of setuptools
- Renamed module from pyhsiclasso to pyhsiclasso
- Remove compatiblility with python 2.
- Update required python version to 3.9
- Switch from str.format() to f-strings
- Moved module code to subdirectory under `src`

[1.13.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.12.0...1.13.0
[1.12.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.11.0...1.12.0
[1.11.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.10.0...1.11.0
[1.10.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.9.0...1.10.0
[1.9.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.8.0...1.9.0
[1.8.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.7.0...1.8.0
[1.7.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.6.0...1.7.0
[1.6.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.5.0...1.6.0
[1.5.0]: https://github.com/olivierlacan/keep-a-changelog/compare/1.5.0
