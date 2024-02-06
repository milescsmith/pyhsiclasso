import importlib.resources as ir
from pyhsiclasso import HSICLasso

stuff = HSICLasso()
thing = ir.files("tests").joinpath("test_data", "csv_data.csv")
stuff.input(thing)