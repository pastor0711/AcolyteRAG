from setuptools import setup

setup(
    name="acolyterag",
    version="0.1.0",
    package_dir={"acolyterag": "."},
    packages=["acolyterag"],
    package_data={
        "acolyterag": ["*.html", "*.css", "*.js"]
    },
    include_package_data=True,
)
