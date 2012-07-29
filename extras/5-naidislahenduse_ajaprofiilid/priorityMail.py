#!/usr/bin/env python
# -*- coding: utf-8 -*-


USELANG = False

import email
import html2text
html2text.UNICODE_SNOB = 1 # No reason to replace unicode characters with ascii lookalikes there
import GeoIP
if USELANG:
    import guess_language
import re
import regexplib
from name2gender import name2gender
import mailbox
import glob
import os.path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
try:
    import cPickle as pickle
except ImportError:
    import pickle

geoip = GeoIP.new(GeoIP.GEOIP_MEMORY_CACHE)
userid = 'andres.erbsen'

words = re.compile(ur'[\wöäüõšž]+',re.UNICODE+re.IGNORECASE)
SEP = u'___'

def getmessagetext_plain(message):
    """ Returns all plaintext content in a message"""
    if message.get_content_type() == 'text/plain':
        encoding = message.get_content_charset()
        text = message.get_payload(decode=True)
        if encoding:
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
    """ Returns all HTML content in a message"""
    if message.get_content_type() == 'text/html':
        encoding = message.get_content_charset()
        text = message.get_payload(decode=True)
        if encoding:
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
    if USELANG:
        language = 'language' +SEP+ guess_language.guessLanguageName(ret)
    for (mark,placeholder) in [(',','comma'),('.','full_stop'),('!','exclaimationmark'),('?','questionmark')]:
        ret = ret.replace(mark, mark+' '+SEP + placeholder+' ')
    if USELANG:
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

def none(*args): return args

def removeshortwords(minlength):
    def f(messagetexts):
        return [ ' '.join(w for w in message.split() if len(w) > minlength) for message in messagetexts]
    return f

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

def messages_from_maildir(maildir):
    for message in maildir:
        message.fp.seek(0)
        yield email.message_from_file(message.fp)

def messages_from_path(path):
    paths = glob.glob(path)
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = glob.glob(os.path.join(path,'*'))
    for filepath in paths:
        try:
            with open(filepath) as f:
                yield email.message_from_file(f)
        except Exception, E:
            print E


def sentby(userid):
    def sentbyuser(message):
        for sender in getheaders(message,'from'):
            if userid in sender:
                return True
    return sentbyuser

def extractfeatures(messages,userid):
    featuretexts = []
    repliedmessageids = set()

    for message in messages:
        if sentby(userid)(message):
            repliedmessageids.update( getheaders(message,'in-reply-to') )
            repliedmessageids.update( getheaders(message,'references') )
        else:
            featuretexts.append( messageinfo(message) )

    def replied_to(messege):
        for message_id in getheaders(message,'message-id'):
            if message_id in repliedmessageids:
                return True
        return False

    isreplied = []
    for message in messages:
        if not sentby(userid)(message):
            r = replied_to(message)
            isreplied.append(r)
    return featuretexts, isreplied

def trainImportanceModel(featuretexts,targetvalues):
    vect = CountVectorizer()
    trf = TfidfTransformer(use_idf=False)
    clf = MultinomialNB(alpha=0.001)

    featurevectors = vect.fit_transform(featuretexts)
    tfvectors = trf.fit_transform(featurevectors)
    clf.fit(tfvectors,targetvalues)

    return (vect,trf,clf)

def predictImportance((vect,trf,clf),featuretexts):
    featurevectors = vect.transform(featuretexts)
    tfvectors = trf.transform(featurevectors)
    predictvector = clf.predict_proba(tfvectors)
    predictlist = predictvector.tolist()
    repliedprobas = [p for (_,p) in predictlist] # the unreplied probability is redundant
    return repliedprobas


USAGE = """usage: priorityMail.py /path/to/messages /path/to/model userid [train]"""

def main():
    args = sys.argv[1:]
    if len(args) == 3:
        args.append('predict')
    if len(args) != 4:
        print USAGE
        return 1

    maildirpath, modelpath, userid, action = args

    try:
        maildir = mailbox.Maildir(maildirpath)
        print 'Reading maildir'
        allmessages = list(messages_from_maildir(maildir))
        print 'Maildir read.'
    except OSError:
        print 'This is not a maildir, so we will hope to just find the messages there'
        allmessages = list(messages_from_path(maildirpath))

    if action == 'train':
        print 'training'
        received_messages_features,replied = extractfeatures(allmessages,userid)
        T = trainImportanceModel(received_messages_features,replied)

        with open(modelpath,'wb') as modelfile:
            pickle.dump(T,modelfile)
    elif action == 'predict':
        print 'predicting'
        with open(modelpath,'rb') as modelfile:
            T = pickle.load(modelfile)

        receivedmessages = filter(lambda m: not sentby(userid)(m),allmessages)
        receivedheadlines = []
        for message in receivedmessages:
            receivedheadlines.append(' '.join(getheaders(message,'subject') + getheaders(message,'from')))
        received_messages_features,alreadyreplied = extractfeatures(allmessages,userid)

        importanceestimate = predictImportance(T,received_messages_features)
        importantmessages = zip(importanceestimate,receivedheadlines)
        importantmessages.sort()
        importantmessages.reverse()
        for (importance,identifier) in importantmessages:
            print identifier.replace('\n',' '), importance

if __name__ == '__main__':
    main()
