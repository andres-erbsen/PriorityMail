#!/usr/bin/env python
# -*- coding: utf-8 -*-

import email
import html2text
html2text.UNICODE_SNOB = 1 # No reason to replace unicode characters with ascii lookalikes there
import GeoIP
import guess_language
import re
import regexplib
import os
import shutil
import datetime

import numpy as np
import pylab as pl
from scipy import interp


import sys
sys.path.append("/opt/sklearn")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm.sparse import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.naive_bayes import MultinomialNB

try:
    import cPickle as pickle
except:
    import pickle

from name2gender import name2gender

encodings = {}

try:
    from IPython.Shell import IPShellEmbed
    embed = IPShellEmbed()
    print 'Embedded shell OK'
except:
    embed = False

import os, errno
import mailbox

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise

geoip = GeoIP.new(GeoIP.GEOIP_MEMORY_CACHE)

words = re.compile(ur'[\wöäüõšž]+',re.UNICODE+re.IGNORECASE)
SEP = u'___'

def getmessagetext_plain(message):
    """ Returns all plaintext content in a message"""
    if message.get_content_type() == 'text/plain':
        encoding = message.get_content_charset()
        text = message.get_payload(decode=True)
        if encoding:
            encodings[encoding] = encodings.get(encoding,0) +1
            text = text.decode(encoding,errors='ignore')
        else:
            # Let's just try to decode it, the chances are this will 
            # work and even a text without unicode characters is better
            # than no text at all
            text = text.decode('unicode-escape',errors='ignore')
        return text + '\n'
    elif message.is_multipart():
        # Parts are message too, so they can consist of parts again. They do.
        return '\n'.join(getmessagetext_plain(part) for part in message.get_payload()).strip('\n')
    else:
        return ''

def getmessagetext_html(message):
    if message.get_content_type() == 'text/html':
        encoding = message.get_content_charset()
        text = message.get_payload(decode=True)
        if encoding:
            encodings[encoding] = encodings.get(encoding,0) +1
            text = text.decode(encoding,errors='ignore')
        else:
            text = text.decode('unicode-escape',errors='ignore')
        try:
            return html2text.html2text(text) + '\n'
        except: # Some html is just invalid...
            return ''
    elif message.is_multipart():
        return '\n'.join(getmessagetext_html(part) for part in message.get_payload()).strip('\n')
    else:
        return ''

def getmessagetext(message):
    """ Extracts text content from email. Parses HTML using html2text if
    no plaintext content is found."""
    if hasattr(message,'fp'): # workaraound for Maildirmessage objects
        message.fp.seek(0)
        message = email.message_from_file(message.fp)
    text = getmessagetext_plain(message)
    if text:
        return text
    return getmessagetext_html(message)

def getcontenttypes(message):
    if hasattr(message,'fp'): # workaraound for Maildirmessage objects
        message.fp.seek(0)
        message = email.message_from_file(message.fp)
    if message.is_multipart():
        return [message.get_content_type()] + sum( (getcontenttypes(part) for part in message.get_payload()),[])
    else:
        return [message.get_content_type()]

def orig(text):
    return '\n'.join( line for line in text.splitlines() if (not line.lstrip().startswith('>')) )

def getheaders(message,header):
    ret = []
    got = message.get_all(header)
    if got:
        for instance in got:
            for text, encoding in email.Header.decode_header(instance):
                if encoding:
                    encodings[encoding] = encodings.get(encoding,0) +1
                    text = text.decode(encoding)
                else:
                    text = text.decode('unicode-escape')
                ret.append(text)
    return ret
    
def getsendergender(fromheader):
    L = fromheader.replace('"','').split()
    L = filter(lambda s: all(c.isalpha() for c in s),L)
    if len(L) > 2:
        L = filter(lambda s: all(all(c == c.lower() for c in ss[1:]) for ss in s.split('-')),L)
    if len(L) > 1:
        L = filter(lambda s: not s.endswith(','),L)
    if len(L) > 2:
        L = filter(lambda s: s[0] == s[0].upper(),L)
    if len(L) > 2:
        filter(name2gender ,L)
    for word in L:
        gender = name2gender(word)
        if gender:
            return gender

def getsenderip(receivedheader):
    # Last address in header is nearest to the sender
    for candidate in reversed(regexplib.ipv4find.findall(receivedheader)):
        if regexplib.ipv4validate.match(candidate):
            return candidate

def getsenderlocation(receivedheader):
    ip = getsenderip(receivedheader)
    if not ip:
        return {}
    ret = dict(country=geoip.country_name_by_addr(ip))
    return ret
    
def messageinfo(message):
    ret = ''
    ret = getmessagetext(message) + '\n\n'
    language = 'language' +SEP+ guess_language.guessLanguageName(ret)
    for (mark,placeholder) in [(',','comma'),('.','full_stop'),('!','exclaimationmark'),('?','questionmark')]:
        ret = ret.replace(mark, mark+' '+SEP + placeholder+' ')
    ret += language + '\n'
    for header in ['subject']: # Headers, that are also content
        ret = ret.rstrip() + '\n'
        for instance in getheaders(message,header):
            ret += instance + ' '
            for word in words.findall(instance):
                ret += header + SEP + word +' '
                ret += word +' '
    ret = ret.rstrip() + '\n'
    headerinfo = set()
    for header in message.keys():
        headerinfo.add('hasheader'+ SEP + header.replace('.','_').replace('-','_'))
    for header in ['sender','to','cc','x-mailer','from','importance','precedence','List-Id']: # ,'sender','to','cc','bcc']:
        for instance in getheaders(message,header):
            instance += ' '+ instance.replace('@','_').replace('.','__')
            if header.startswith('x-'): header = header[2:]
            for word in words.findall(instance):
                if sum(c.isalpha() for c in word) > (len(word)/3*2):
                    headerinfo.add(header + SEP + word)
    receivedheaders = '\n'.join(getheaders(message,'received'))
    if getsenderip(receivedheaders):
        headerinfo.add('from_ip'+ SEP + getsenderip(receivedheaders).replace('.','_'))
    for k,v in getsenderlocation(receivedheaders).iteritems():
        if v:
            headerinfo.add(u'from_location_'+k+ SEP + v.decode('utf-8').replace(' ','_'))
    gender = getsendergender('\n'.join(getheaders(message,'from')))
    headerinfo.add('from_gender' +SEP +str(gender))
    for contenttype in getcontenttypes(message):
        headerinfo.add('contains' + SEP + contenttype.replace('/','_'))
    return ret+'\n'+' '.join(headerinfo)

def showmessage(id):
    if type(id) == type(1234):
        message = messages[id]
    else:
        raise ValueError
    for header in ['from','sender','to','cc']:
        print header+':',
        for instance in getheaders(message,header):
            if instance != 'None':
                print instance,
        print
    print getmessagetext(message)

def none(*args): return args

def doubleapply(f):
    def g(a,b):
        return f(a),f(b)
    return g

def removeshortwords(minlength):
    def f(messagetexts):
        return [ ' '.join(w for w in message.split() if len(w) > minlength) for message in messagetexts]
    return doubleapply(f)

@doubleapply
def textonly(messagetexts):
    return [ ' '.join(w for w in message.split() if SEP not in w) for message in messagetexts]

@doubleapply
def nolanguage(messagetexts):
    return [ ' '.join(w for w in message.split() if ('language'+SEP) not in w) for message in messagetexts]

def tf(train,test):
    trf = TfidfTransformer(use_idf=False)
    trf = trf.fit(train)
    train = trf.transform(train)
    test = trf.transform(test)
    return train,test

def tfidf(train,test):
    trf = TfidfTransformer()
    trf = trf.fit(train)
    train = trf.transform(train)
    test = trf.transform(test)
    return train,test

classifiers = [('MNB',MultinomialNB(alpha=0.001)),
           ('linearSVC',SVC(kernel='linear',C=2,probability=True)),
#           ('polySVC',SVC(kernel='poly',C=2^7,degree=2,probability=True)),
#           ('sigmoidSVC',SVC(kernel='sigmoid',C=0.5,probability=True)),
           ('rbfSVC',SVC(kernel='rbf',C=4,gamma=1,probability=True)),
           ]

postvects = [('Counts',none),('TF',tf),('TFIDF',tfidf)]

classificationmethods = []
for clfn,clf in classifiers:
    for pvn,pv in postvects:
        classificationmethods.append(('-'.join((pvn,clfn)),none,CountVectorizer(),pv,clf))

classificationmethods = [(('MNB alpha=%0.4f' % alpha), none, CountVectorizer(), tf, MultinomialNB(alpha=alpha)) for alpha in [1.,0.5,0.25,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]]

print classificationmethods

maildir = mailbox.Maildir('/home/andres/mail/andres.erbsen@gmail.com/all_mail')
userid = 'andres.erbsen'
messages = []
messages_as_text = []
repliedmessageids = set()

nmessages = 1000000
nfolds = 10

for i,message in enumerate(maildir):
    if i >= nmessages: break
    print ('Parsing message %d: %s' % (i+1,
        str(message.getheader('subject')).replace('\n','')))[:80]
    if userid in message.getheader('from'):
        repliedmessageids.update( message.getheaders('in-reply-to') )
        repliedmessageids.update( message.getheaders('references') )
    else:
        message.fp.seek(0)
        message = email.message_from_file(message.fp)
        messages.append(message)
        messages_as_text.append( messageinfo(message) )

# isreplied = [ message.getheader('message-id') in repliedmessageids for message in maildir if userid not in message.getheader('from') ]
isreplied = []
for i,message in enumerate(maildir):
    if i >= nmessages: break
    if userid not in message.getheader('from'):
        repl = (message.getheader('message-id') in repliedmessageids)
        isreplied.append(repl)

y = target = np.array(isreplied)
cv = StratifiedKFold(target, k=nfolds)

# dict of {name:[plots for each run]}
prrecplots = {}
rocplots = {}

for name,prevect,vect,postvect,clf in classificationmethods:
    for k, (train, test) in enumerate(cv):
        print name,'run',1+k
        trainmessages = [m for i,m in enumerate(messages_as_text) if i in train]
        testmessages = [m for i,m in enumerate(messages_as_text) if i in test]
        print 'Training using %d messages and testing using %d messages' % (len(trainmessages),len(testmessages))
        trainmessages,testmessages = prevect(trainmessages,testmessages)
        traindata = vect.fit_transform(trainmessages)
        testdata = vect.transform(testmessages)
        traindata,testdata = postvect(traindata,testdata)
        traintarget = np.asarray([y[i] for i in train])
        testtarget = np.asarray([y[i] for i in test])
        print traindata.shape, traintarget.shape
        if isinstance(clf,SVC) and clf.kernel == 'linear':
            clf.fit(traindata,traintarget,class_weight='auto')
        else:
            clf.fit(traindata,traintarget)
        probas = clf.predict_proba(testdata)
        rocdata = roc_curve(testtarget, probas[:,1])
        rocplots[name] = rocplots.get(name,[]) + [rocdata]
        prrecdata = precision_recall_curve(testtarget, probas[:,1])
        prrecplots[name] = prrecplots.get(name,[]) + [prrecdata]
        gotprobabs = True


plotdir = os.path.abspath(datetime.datetime.today().strftime("%Y-%m-%d_%H-%M"))
mkdir_p(plotdir)
with open(os.path.join(plotdir,'plot.pc'),'w') as f:
    pickle.dump({'rocplots':rocplots,'prrecplots':prrecplots},f)
shutil.copy2(os.path.abspath(sys.argv[0]), os.path.join(plotdir,os.path.split(sys.argv[0])[-1]))

meanrocplots = []
for name in rocplots:
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i,(fpr,tpr,thresholds) in enumerate(rocplots[name]):
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        tpr[0] = 0.0
        tpr[-1] = 1.0
        roc_auc = auc(fpr, tpr)
        pl.plot(fpr, tpr, lw=1, label='fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(rocplots[name])
    mean_tpr[-1] = 1.0
    meanrocplots.append((name,mean_tpr,mean_fpr))
    # also put the mean on the folds plot
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr,
            label=name+' (area = %0.2f)' % mean_auc, lw=2)
    # folds plot details
    pl.plot([0, 1], [0, 1], '--', color=(0.6,0.6,0.6))
    pl.title('%s receiver operating characteristic per fold'%name)
    pl.ylabel('True positive rate')
    pl.xlabel('False positive rate')
    pl.legend(loc="lower right")
    pl.savefig(os.path.join(plotdir,name.encode('ascii',errors='ignore')+'.roc.svg'))
    pl.close()

for (name,mean_tpr,mean_fpr) in meanrocplots:
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr,
            label=name+' (area = %0.2f)' % mean_auc, lw=2)

pl.plot([0, 1], [0, 1], '--', color=(0.6,0.6,0.6))
pl.title('Receiver operating characteristic')
pl.ylabel('True positive rate')
pl.xlabel('False positive rate')
pl.legend(loc="lower right")
pl.savefig(os.path.join(plotdir,'meanroc'.encode('ascii',errors='ignore')+'.roc.svg'))
pl.close()


meanprrecplots = [] 
for name in prrecplots:
    mean_recall = 0.0
    mean_precision = np.linspace(0, 1, 100)
    for i,(precision,recall,thresholds) in enumerate(prrecplots[name]):
        mean_recall += interp(mean_precision, precision, recall)
        recall[-1] = 0.0
        prrec_auc = auc(precision, recall)
        pl.plot(precision, recall, lw=1, label='fold %d (area = %0.2f)' % (i, prrec_auc))
    mean_recall /= len(prrecplots[name])
    mean_recall[-1] = 0.0
    meanprrecplots.append((name,mean_recall,mean_precision))
    mean_auc = auc(mean_precision, mean_recall)
    # also put the mean on the folds plot
    mean_auc = auc(mean_precision, mean_recall)
    pl.plot(mean_precision, mean_recall,
            label=name+' (area = %0.2f)' % mean_auc, lw=2)
    # folds plot details
    pl.title('%s precision vs recall per fold'%name)
    pl.ylabel('Recall')
    pl.xlabel('Precision')
    pl.legend(loc="upper right")
    pl.savefig(os.path.join(plotdir,name.encode('ascii',errors='ignore')+'.prrec.svg'))
    pl.close()

for (name,mean_recall,mean_precision) in meanprrecplots:
    mean_auc = auc(mean_precision, mean_recall)
    pl.plot(mean_precision, mean_recall,
            label=name+' (area = %0.2f)' % mean_auc, lw=2)

pl.title('Precision vs recall')
pl.ylabel('Recall')
pl.xlabel('Precision')
pl.legend(loc="upper right")
pl.savefig(os.path.join(plotdir,'meanprrec'.encode('ascii',errors='ignore')+'.prrec.svg'))
pl.close()

with open(os.path.join(plotdir,'meanplot.pc'),'w') as f:
    pickle.dump({'meanrocplots':meanrocplots,'meanprrecplots':meanprrecplots},f)

#~ if embed: embed()
