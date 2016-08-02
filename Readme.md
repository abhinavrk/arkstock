This is a finance package built on top of yahoo finance meant for research purposes, particularly dealing with the problem of diversification and portfolio optimization. 

**Note: Yahoo Finance does not allow their APIs to be used for commercial purposes. Since (as of Aug 1st 2016) this package is built on yahoo finance, this means that you cannot use the code base as is for commercial purposes.**

The documentation is available (generated via pdoc) as html files and inline comments, along with copious type hints.

For a quick hint into how to use this package, I would suggest that you look into `integ_test.py` which provides a simple one line example on how to use this package.

Package dependancies include (but are not limited to):

* `numpy`
* `scipy`
* `scikit-learn`
* `python 3.5`

The goal for this package/project was to experiment with different implementations to come up with a simple elegant way to quickly diversify a portfolio. The methods implemented within this package are not tried or tested and you may rely on them at your own risk. As I've mentioned before, this was simply for research purposes.

In order to allow this to grow, I've tried to make it easy to add in alternate data sources and alternate algorithms. These can all be tied together by simply extending `PortfolioGeneratorFactory` in `pipelines.py`. Please see the code for more details.

I had a lot of interesting discussions regarding the viability of the algorithms with Seyon Vasantharajan and Emil Kerimov and would like to take this opportunity to thank them. I also read a wide array of notes regarding MVO and PCA.

The code is released under the MIT licence. Feel free to fork and modify. Happy coding!