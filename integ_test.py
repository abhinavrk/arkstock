import sys
sys.dont_write_bytecode = True
from arkstock import PortfolioGeneratorFactory, YahooFinanceMetadataClient, YahooFinanceClient
from arkstock import yahoo_finance_gateway as yfgateway

import time
portfolio_opt = PortfolioGeneratorFactory.get_default(lambda x: x in ('TOR', 'NMS'))

start = time.time()
warmup = portfolio_opt([])
end = time.time()
print('first run ', end - start)

start = time.time()
portfolio_opt(['MDA.TO'])
portfolio_opt(['GOOGL'])
portfolio_opt(['GTE.TO'])
end = time.time()
print('non-warmup runs ', (end - start)/3.)