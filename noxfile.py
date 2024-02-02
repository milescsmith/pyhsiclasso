import nox
from nox_poetry import session

package = "pyhsiclasso"
# nox.options.sessions = ["lint", "black", "tests"]
nox.options.sessions = ["tests"]
locations = "src", "tests", "noxfile.py", "docs/conf.py"

@session(python="3.10")
def tests(session: session) -> None:
    # args = session.posargs or locations
    # install_with_constraints(session, ".")
    # install_with_constraints(session, "pytest")
    session.run_always("poetry", "install", external=True)
    session.run("pytest")


@session()
def black(session: session) -> None:
    """Run black code formatter."""
    session.install("black")
    session.run_always("poetry", "install", external=True)
    session.run("black", "src")


@session
def lint(session: session) -> None:
    """Lint using ruff."""
    args = session.posargs or locations
    session.install("ruff")
    session.run("ruff", *args)
