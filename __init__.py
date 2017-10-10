"""
Machine learning module for Python
==================================


zz is a Python module integrating feature engineering and
some model ensemble methods in the packages.


It aims to provide simple and efficient solutions to machine
learning problems.


Authors: LI AO <aoli@hit.edu.cn>
"""
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
handler.setFormatter(logging.Formatter(formatter))
logger.addHandler(handler)

__version__ = '0.0.0-2017-10-8'

__all__ = ['featrues', 'models']
