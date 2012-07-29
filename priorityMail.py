#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Email is a mess. This program is not intended to understand every
# possible encoding or content type and will probably fail to understand
# some messages, in particular HTML.

import email
import re
import mailbox
import glob
import os.path
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
try:
    import cPickle as pickle
except ImportError:
    import pickle

LINEW = 80 # line width of output
words = re.compile(ur'[\wöäüõšž]+',re.UNICODE+re.IGNORECASE)
SEP = u'___'

def getmessagetext_plain(message):
    """Returns all plaintext content in a message"""
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

def getmessagetext(message):
    """Extracts text content from email."""
    if hasattr(message,'fp'): # workaraound for Maildirmessage objects
        message.fp.seek(0)
        message = email.message_from_file(message.fp)
    return getmessagetext_plain(message)

def getcontenttypes(message):
    """List all content types in a message"""
    if hasattr(message,'fp'): # workaraound for Maildirmessage objects
        message.fp.seek(0)
        message = email.message_from_file(message.fp)
    if message.is_multipart():
        return [message.get_content_type()] + sum( (getcontenttypes(part) for part in message.get_payload()),[])
    else:
        return [message.get_content_type()]

def orig(text):
    """ Remove ">"-quotes from a text"""
    return '\n'.join(
        line for line in text.splitlines() 
        if (not line.lstrip().startswith('>'))
    )

def getheaders(message,header):
    ret = []
    got = message.get_all(header)
    if got:
        for instance in got:
            for text, encoding in email.Header.decode_header(instance):
                if encoding:
                    text = text.decode(encoding,errors='ignore')
                else:
                    text = text.decode('unicode-escape',errors='ignore')
                ret.append(text)
    return ret
    

def messageinfo(message):
    ret = ''
    ret = getmessagetext(message) + '\n\n'
    for (mark,placeholder) in [(',','comma'),('.','full_stop'),('!','exclaimationmark'),('?','questionmark')]:
        ret = ret.replace(mark, mark+' '+SEP + placeholder+' ')
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
                # Heuristic to filter out numeric garbage
                if sum(c.isalpha() for c in word) > (len(word)/3*2):
                    headerinfo.add(header + SEP + word)
    for contenttype in getcontenttypes(message):
        headerinfo.add('contains' + SEP + contenttype.replace('/','_'))
    return ret+'\n'+' '.join(headerinfo)

def none(*args):
    """a variant of the identity function"""
    return args

def removeshortwords(minlength):
    """Returns a function that removes all words shorter than @minlength
    from all @messagetexts passed to it"""
    def f(messagetexts):
        return [ ' '.join(w for w in message.split() if len(w) > minlength) for message in messagetexts]
    return f

def tf(train,test):
    """Transform feature vectors: TF"""
    trf = TfidfTransformer(use_idf=False)
    trf = trf.fit(train)
    train = trf.transform(train)
    test = trf.transform(test)
    return train,test

def tfidf(train,test):
    """Transform feature vectors: TFIDF"""
    trf = TfidfTransformer()
    trf = trf.fit(train)
    train = trf.transform(train)
    test = trf.transform(test)
    return train,test

def messages_from_maildir(maildir):
    """ Iterate over all email messages in a maildir"""
    for message in maildir:
        message.fp.seek(0)
        yield email.message_from_file(message.fp)

def messages_from_path(path):
    """ Iterate over all email messages in a (globbable) path"""
    paths = glob.glob(path)
    if len(paths) == 1 and os.path.isdir(paths[0]):
        paths = glob.glob(os.path.join(path,'*'))
    for filepath in paths:
        filepath = os.path.expanduser(filepath)
        try:
            with open(filepath) as f:
                yield email.message_from_file(f)
        except Exception, E:
            print E


def sentby(userid):
    """ Returns a function that tests if a message is sent by @userid"""
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
            # In a real application one would handle all funny encodings
            # and be concerned of every error, but let's make it simple:
            # every message that our application cannot decode is empty
            try:
                featuretexts.append( messageinfo(message) )
            except:
                print ("Failed to parse message. Sorry.")
                featuretexts.append( '' )

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


USAGE = """usage: olulisedkirjad.py /path/to/messages /path/to/model userid [train]"""

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
        # force evaluation. Probably a python maildir bug otherwise.
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
        received_messages_features,alreadyreplied = extractfeatures(allmessages,userid)

        importanceestimate = predictImportance(T,received_messages_features)
        importantmessages = zip(importanceestimate,receivedmessages)
        importantmessages.sort()
        importantmessages.reverse()

        for (importance,message) in importantmessages:
            headline = ''
            fromheader = ' '.join(getheaders(message,'from'))
            headline += ' '.join(re.findall('<.*>',fromheader))[1:-1].strip()
            headline += " | "
            headline += ' '.join(getheaders(message,'subject'))
            headline = headline.replace('\n',' ')
            headline = headline[:LINEW-11] # room for importance
            headline += " | "
            headline += ' '*(LINEW-8 - len(headline))
            headline += "%0.3g" % importance
            print headline

if __name__ == '__main__':
    main()
