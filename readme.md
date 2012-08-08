This software originally accompanied my term paper in Tallinn Secondary Science
School. The paper (in Estonian) is included, but in accordance to school rules is also
overwhelmingly verbose. In case somebody actually would want to read it, I'd advise
them to skip to chapter 4.

# Dependencies
Python 2.7 and scikit-learn 0.10. There is a more bloated version in the extras
directory. Tested on Ubuntu and Arch Linux, reported to work on Windows 7.

# Abstract
Email is one of the most common means of electronic communication. In many
cases, handling substantial amounts of messages on a day-by-day basis can be a
hassle, especially if the time available for this is scarce. 
A possible solution to increase productivity in such situations would be a
system of automatic classification or ranking of received messages, which could
detect important messages and distinguish them from other kinds of
communication, for example, list mail. 
This paper investigates the feasibility of such automatic classification based
on the content of the messages previously received and possibly replied to by
the same user. Email text is used as the main source of features for
classification, additional features extracted from headers and generalisations
on these are assessed. Building on prior successes in spam detection, well-know
classifiers such as multinomial naive Bayes and support vector machines (with
linear and RBF kernels) are used to classify TFIDF, TF or feature counts
representations of the messages. The performances of the combinations of these
primitives are compared using ROC and precision-recall curves. With a false
positive rate of 0.15 a true positive rate of 0.9 is achieved using multinomial
naive Bayes with TF feature vectors. Considering the fact that not all important
messages are replied to, this is a fairly good result, which should be useful,
as described in the beginning of this paragraph. Therefore, a minimal example
application to fit a model using the messages in local folders to classify new
ones is created.

# Licence
AGPL3 for software and CC BY-SA 3.0 for everything else.
