#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Setup script.

Authors: wangfan04(wangfan04@baidu.com)
Date:    2019/05/22 19:30:16
"""

import setuptools

setuptools.setup(
  name="intrabuildingtransport",  #pypi中的名称，pip或者easy_install安装时使用的名称
  version="1.0",
  author="",
  author_email="",
  description=("A reinforcement learning benchmark elevators"),
  license="GPLv3",
  keywords="redis subscripe",
  url="",
  packages=['intrabuildingtransport'])  # 需要打包的目录列表
  #install_requires=['parl>=1.1'])
