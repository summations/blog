#!/usr/bin/env python
# -*- coding: utf-8 -*- #

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

import os
import sys

# import main config
sys.path.append(os.curdir)
from pelicanconf import *

# Use real live URLs
SITESCHEME = 'http'
SITEURL = SITESCHEME + '://summations.github.io'
RELATIVE_URLS = False

#
# Generate Feeds
#
# The domain prepended to feed URLs. Since feed URLs should always be absolute,
# it is highly recommended to define this.
FEED_DOMAIN = SITEURL

FEED_ATOM = 'feeds/all.atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'

FEED_RSS = 'feeds/all.rss.xml'
CATEGORY_FEED_RSS = 'feeds/%s.rss.xml'

# Always start over
DELETE_OUTPUT_DIRECTORY = False

# Articles have comments
DISQUS_SITENAME = "rahulpathakml"

# ARTICLE_TWEET_BUTTON = True
# ARTICLE_GOOGLEPLUS_BUTTON = True

# WITH_FUTURE_DATES = True
# CACHE_CONTENT = False

#################################
#
# Setting for plugins
#
#################################

# Which extra plugins to enable when publishing
PLUGINS = PLUGINS + ['gzip_cache', 'sitemap']

# Settings for the sitemap plugin
SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.8,
        'indexes': 0.7,
        'pages': 0.5
    },
    'changefreqs': {
        'articles': 'weekly',
        'indexes': 'daily',
        'pages': 'monthly'
    }
}
