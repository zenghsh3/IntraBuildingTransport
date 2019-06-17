# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m intrabuildingtransport方式直接执行。

Authors: wangfan04(wangfan04@baidu.com)
Date:    2019/05/22 19:30:16
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from intrabuildingtransport.cmdline import main
sys.exit(main())
