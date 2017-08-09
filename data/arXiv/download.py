#!/usr/local/bin/python2.7
# downlaod arXiv metadata
# A improved version of http://betatim.github.io/posts/analysing-the-arxiv/
# works perfect in jupyter notebook. but poor performance by directly reunning this script.
# So please copy the code into a jupyter notebook to run it.
# by the way, arXiv's API can't be more diffifult to use.
# They close their serve for maintainence every 7:30 - 8:30 (UTC+8)

import os
import time
import urllib2
import datetime
from itertools import ifilter
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import bibtexparser

pd.set_option('mode.chained_assignment','warn')

%matplotlib inline

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
data_save_path = './data/'

def dl(f, u):
    df = pd.DataFrame(columns=("title", "abstract", "categories", "created", "id", "doi"))
    base_url = "http://export.arxiv.org/oai2?verb=ListRecords&"
    url = (base_url +
           "from=" + f + "&until=" + u +"&metadataPrefix=arXiv")

    while True:
        print "fetching", url
        try:
            response = urllib2.urlopen(url)

        except urllib2.HTTPError, e:
            if e.code == 503:
                to = int(e.hdrs.get("retry-after", 30))
                to = 10
                print "Got 503. Retrying after {0:d} seconds.".format(to)
                time.sleep(to)
                continue

            else:
                raise

        xml = response.read()

        root = ET.fromstring(xml)

        for record in root.find(OAI+'ListRecords').findall(OAI+"record"):
            arxiv_id = record.find(OAI+'header').find(OAI+'identifier')
            meta = record.find(OAI+'metadata')
            try:
                info = meta.find(ARXIV+"arXiv")
            except:
                continue
            created = info.find(ARXIV+"created").text
            created = datetime.datetime.strptime(created, "%Y-%m-%d")
            categories = info.find(ARXIV+"categories").text

            # if there is more than one DOI use the first one
            # often the second one (if it exists at all) refers
            # to an eratum or similar
            doi = info.find(ARXIV+"doi")
            if doi is not None:
                doi = doi.text.split()[0]

            contents = {'title': info.find(ARXIV+"title").text,
                        'id': info.find(ARXIV+"id").text,#arxiv_id.text[4:],
                        'abstract': info.find(ARXIV+"abstract").text.strip(),
                        'created': created,
                        'categories': categories.split(),
                        'doi': doi,
                        }

            df = df.append(contents, ignore_index=True)

        # The list of articles returned by the API comes in chunks of
        # 1000 articles. The presence of a resumptionToken tells us that
        # there is more to be fetched.
        token = root.find(OAI+'ListRecords').find(OAI+"resumptionToken")
        if token is None or token.text is None:
            break

        else:
            url = base_url + "resumptionToken=%s"%(token.text)
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    filename = data_save_path + f + '_' + u + '.csv'
    try:
        df.to_csv(filename)
    except:
        df.to_csv(filename, encoding='utf-8')
    print "saving to " + filename
    os.system('say "finished"')

# TODO: FIX THIS!!! THIS IS STUPID!!!

dl('2017-07-01', '2017-07-13')
dl('2017-06-01', '2017-06-31')
dl('2017-05-01', '2017-05-31')
dl('2017-04-01', '2017-04-31')
dl('2017-03-01', '2017-03-31')
dl('2017-02-01', '2017-02-31')
dl('2017-01-01', '2017-01-31')
dl('2016-12-01', '2016-12-31')
dl('2016-11-01', '2016-11-31')
dl('2016-10-01', '2016-10-31')
dl('2016-09-01', '2016-09-31')
dl('2016-08-01', '2016-08-31')
dl('2016-07-01', '2016-07-31')
dl('2016-06-01', '2016-06-31')
dl('2016-05-01', '2016-05-31')
dl('2016-04-01', '2016-04-31')
dl('2016-03-01', '2016-03-31')
dl('2016-02-01', '2016-02-31')
dl('2016-01-01', '2016-01-31')
dl('2015-12-01', '2015-12-31')
dl('2015-11-01', '2015-11-31')
dl('2015-10-01', '2015-10-31')
dl('2015-09-01', '2015-09-31')
dl('2015-08-01', '2015-08-31')
dl('2015-07-01', '2015-07-31')
dl('2015-06-01', '2015-06-31')
dl('2015-05-01', '2015-05-31')
dl('2015-04-01', '2015-04-31')
dl('2015-03-01', '2015-03-31')
dl('2015-02-01', '2015-02-31')
dl('2015-01-01', '2015-01-31')
dl('2014-12-01', '2014-12-31')
dl('2014-11-01', '2014-11-31')
dl('2014-10-01', '2014-10-31')
dl('2014-09-01', '2014-09-31')
dl('2014-08-01', '2014-08-31')
dl('2014-07-01', '2014-07-31')
dl('2014-06-01', '2014-06-31')
dl('2014-05-01', '2014-05-31')
dl('2014-04-01', '2014-04-31')
dl('2014-03-01', '2014-03-31')
dl('2014-02-01', '2014-02-31')
dl('2014-01-01', '2014-01-31')
dl('2013-12-01', '2013-12-31')
dl('2013-11-01', '2013-11-31')
dl('2013-10-01', '2013-10-31')
dl('2013-09-01', '2013-09-31')
dl('2013-08-01', '2013-08-31')
dl('2013-07-01', '2013-07-31')
dl('2013-06-01', '2013-06-31')
dl('2013-05-01', '2013-05-31')
dl('2013-04-01', '2013-04-31')
dl('2013-03-01', '2013-03-31')
dl('2013-02-01', '2013-02-31')
dl('2013-01-01', '2013-01-31')
dl('2012-12-01', '2012-12-31')
dl('2012-11-01', '2012-11-31')
dl('2012-10-01', '2012-10-31')
dl('2012-09-01', '2012-09-31')
dl('2012-08-01', '2012-08-31')
dl('2012-07-01', '2012-07-31')
dl('2012-06-01', '2012-06-31')
dl('2012-05-01', '2012-05-31')
dl('2012-04-01', '2012-04-31')
dl('2012-03-01', '2012-03-31')
dl('2012-02-01', '2012-02-31')
dl('2012-01-01', '2012-01-31')
dl('2011-12-01', '2011-12-31')
dl('2011-11-01', '2011-11-31')
dl('2011-10-01', '2011-10-31')
dl('2011-09-01', '2011-09-31')
dl('2011-08-01', '2011-08-31')
dl('2011-07-01', '2011-07-31')
dl('2011-06-01', '2011-06-31')
dl('2011-05-01', '2011-05-31')
dl('2011-04-01', '2011-04-31')
dl('2011-03-01', '2011-03-31')
dl('2011-02-01', '2011-02-31')
dl('2011-01-01', '2011-01-31')
dl('2010-12-01', '2010-12-31')
dl('2010-11-01', '2010-11-31')
dl('2010-10-01', '2010-10-31')
dl('2010-09-01', '2010-09-31')
dl('2010-08-01', '2010-08-31')
dl('2010-07-01', '2010-07-31')
dl('2010-06-01', '2010-06-31')
dl('2010-05-01', '2010-05-31')
dl('2010-04-01', '2010-04-31')
dl('2010-03-01', '2010-03-31')
dl('2010-02-01', '2010-02-31')
dl('2010-01-01', '2010-01-31')
dl('2009-12-01', '2009-12-31')
dl('2009-11-01', '2009-11-31')
dl('2009-10-01', '2009-10-31')
dl('2009-09-01', '2009-09-31')
dl('2009-08-01', '2009-08-31')
dl('2009-07-01', '2009-07-31')
dl('2009-06-01', '2009-06-31')
dl('2009-05-01', '2009-05-31')
dl('2009-04-01', '2009-04-31')
dl('2009-03-01', '2009-03-31')
dl('2009-02-01', '2009-02-31')
dl('2009-01-01', '2009-01-31')
dl('2008-12-01', '2008-12-31')
dl('2008-11-01', '2008-11-31')
dl('2008-10-01', '2008-10-31')
dl('2008-09-01', '2008-09-31')
dl('2008-08-01', '2008-08-31')
dl('2008-07-01', '2008-07-31')
dl('2008-06-01', '2008-06-31')
dl('2008-05-01', '2008-05-31')
dl('2008-04-01', '2008-04-31')
dl('2008-03-01', '2008-03-31')
dl('2008-02-01', '2008-02-31')
dl('2008-01-01', '2008-01-31')
dl('2007-12-01', '2007-12-31')
dl('2007-11-01', '2007-11-31')
dl('2007-10-01', '2007-10-31')
dl('2007-09-01', '2007-09-31')
dl('2007-08-01', '2007-08-31')
dl('2007-07-01', '2007-07-31')
dl('2007-06-01', '2007-06-31')
dl('2007-05-01', '2007-05-31')

