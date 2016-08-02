import sys
sys.dont_write_bytecode = True

import unittest

from arkstock import testclasses

tests = __import__('arkstock', globals(), locals(), testclasses, 0)

if __name__ == '__main__':
    unittest.main(tests)