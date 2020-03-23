class StopWords():
    stopWordsMerged = ["hasn't", 'see', 'own', 'other', 'went', 'made', 'much', 'present', 'like', "they're", 'knew', 'here', 'were', 'from', 'in', "aren't", 'furthers', 'latest', 'small', 'what', 'n', 'our',
                       "how's", "she's", 'next', 'who', 'will', 'areas', 'y', 'mostly', "wasn't", 'new', 'ought', "where's", "mustn't", 'don', 'q', 'several', 'something', 'whether', 'problems', 'said', 'work',
                       'such', 'anything', 'opening', 'i', "it's", 'orders', "he'd", 'necessary', 'seemed', 'though', 'oldest', 'keeps', 'make', 'number', 'since', 'o', 'same', "isn't", 'states', "shouldn't",
                       'ways', 'gave', 'smallest', 'part', "you've", 'presents', 'used', 'your', 'by', "didn't", 'is', 'differently', 'end', 'get', 'ends', 'my', 'would', 'really', 'enough', 'got', 'myself',
                       'v', 'others', 'going', "hadn't", 'face', 'has', 'interest', 'second', 'ours', 'a', 'be', 'per', 'longer', 'every', 'member', 'wanting', 'differ', "we'll", 'sides', 'given', 'can', 'non',
                       'might', 'three', 'possible', 'state', 'had', 'certain', 'then', 'finds', "here's", 'ordering', 'turns', 'although', "she'd", "they've", "he'll", 'l', 'newest', 'cannot', 'backs', "they'll",
                       'faces', 'know', 'it', 'g', 'each', 'just', 'kind', 'even', "i'll", 'above', 'problem', 'likely', 'about', 'thinks', 'them', 'this', 'whose', 'asks', 'up', 'room', 'already', 'parts',
                       'open', 'early', 'both', 'two', 'take', 'place', 'toward', "we'd", 'u', 'around', 'facts', "we've", 'always', 'b', 'interested', 'under', 'making', 'put', 'points', 'does', 'at', 'there',
                       'upon', 'and', 'too', 'once', 'another', 'one', 'become', 'point', 'things', 'do', "shan't", 'clearly', "there's", 'been', 'years', 'w', 'perhaps', 'noone', 'only', 'when', 'evenly',
                       'pointing', 'later', "what's", 'ordered', 'whom', 'out', 'below', 'interesting', 'clear', 'an', 'turning', 'or', "let's", 'began', 'took', 'however', 'presenting', 'behind', 'find',
                       'grouped', 'yourselves', 't', 'seem', 'anywhere', 'k', 'being', 'many', 'h', 'ask', 'having', 'came', 'felt', 'themselves', 'parting', 'therefore', 'nor', 'd', "when's", 'needing',
                       'was', 'seconds', 'area', 'me', "she'll", 'thoughts', 'come', 'opened', 'whole', 'very', "you'll", 'older', 'wants', 'great', 'itself', 'working', 'across', 'because', 'we', 'herself',
                       "doesn't", 'cases', 'everywhere', 'sees', 'highest', 'few', 'show', 'mrs', 'those', 'somebody', 'man', 'first', 'saw', "why's", 'most', 'nowhere', 'group', 'on', 'give', 'almost',
                       'some', 'became', "can't", 'greater', 'why', 'c', 'again', 'asking', 'young', 'generally', 'furthered', 'well', "he's", 'without', 'f', 'downs', 'so', "that's", 'along', 'anyone',
                       'showed', 'greatest', 'somewhere', 'have', 'known', 'turn', 'someone', 'use', 'uses', 'how', 'until', 'sure', 'far', "i'm", 'furthering', 'against', 'done', 'which', 'thus', 'numbers',
                       'him', 'he', 'puts', 'long', 'am', 'seeming', 'where', 'wells', "they'd", 'presented', "who's", 'least', 'more', 'she', 'between', 'younger', 'says', 'taken', "weren't", 'becomes',
                       "we're", 'gives', 'rather', 'quite', 'knows', 'any', 'longest', 's', 'but', 'her', 'everything', 'also', 'four', 'parted', 'everybody', 'theirs', 'off', 'are', 'lets', 'backing',
                       'the', "couldn't", 'ending', 'ever', 'before', 'shows', 'with', 'among', 'year', 'hers', "i'd", 'showing', 'into', "wouldn't", 'of', 'large', 'way', 'fact', 'if', 'often', 'nothing',
                       'downed', 'after', 'everyone', 'mr', 'say', 'needs', 'as', 'asked', "you'd", 'to', 'grouping', 'must', 'doing', 'did', 'over', 'away', 'his', 'groups', 'men', 'now', 'never', 'last',
                       'through', 'p', 'side', 'back', 'while', 'largely', 'pointed', 'r', 'nobody', 'e', 'keep', 'could', 'important', 'newer', 'smaller', 'wanted', 'still', 'together', "i've", 'seems',
                       'backed', 'opens', 'for', "don't", 'beings', 'ourselves', 'down', 'want', 'fully', 'different', 'its', 'higher', 'right', 'within', 'think', 'gets', 'x', 'youngest', 'that', "won't",
                       'they', 'old', 'anybody', 'works', 'less', 'during', 'turned', 'big', 'no', 'alone', 'today', 'j', 'worked', 'further', 'places', 'yours', 'case', 'all', 'downing', 'yourself',
                       'order', 'may', 'himself', 'their', 'need', 'members', 'z', 'full', 'needed', 'goods', 'm', 'interests', 'rooms', 'these', 'high', 'let', 'thought', 'not', 'yet', 'go', 'certainly',
                       'general', 'should', 'than', 'you', 'either', "haven't", 'us', 'shall', "you're", 'ended', 'thing']

    stopWordsSpark = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                      'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                      'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                      'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                      'can', 'will', 'just', 'don', 'should', 'now', "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'm", "you're", "he's", "she's",
                      "it's", "we're", "they're", "i've", "we've", "you've", "they've", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "don't", "doesn't", "didn't", "won't", "wouldn't",
                      "shan't", "shouldn't", "mustn't", "can't", "couldn't", 'cannot', 'could', "here's", "how's", "let's", 'ought', "that's", "there's", "what's", "when's", "where's", "who's", "why's", 'would']

    stopWordsKNIME = ['been', 'mostly', 'year', 'areas', 'your', 'without', 'these', 'finds', 'would', 'because', 'you', 'sure', 'thus', 'going', 'younger', 'an', 'whose', 'as', 'at', 'turning', 'downed',
                      'much', 'be', 'anybody', 'ordering', 'least', 'turn', 'how', 'see', 'same', 'by', 'after', 'a', 'ordered', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'possible', 'right', 'l',
                      'm', 'n', 'o', 'newer', 'p', 'the', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'faces', 'under', 'yours', 'did', 'rooms', 'do', 'down', 'got', 'later', 'others', 'area', 'needs',
                      'which', 'making', 'need', 'its', 'often', 'never', 'she', 'take', 'parts', 'therefore', 'however', 'some', 'furthers', 'rather', 'young', 'smallest', 'for', 'back', 'states', 'nowhere',
                      'perhaps', 'end', 'just', 'over', 'go', 'room', 'with', 'although', 'there', 'well', 'he', 'showing', 'presents', 'big', 'very', 'smaller', 'wanting', 'years', 'turns', 'number', 'four',
                      'per', 'if', 'order', 'likely', 'went', 'large', 'in', 'made', 'nothing', 'is', 'being', 'it', 'somebody', 'ever', 'even', 'asks', 'gave', 'opens', 'orders', 'presenting', 'become', 'other',
                      'longest', 'works', 'turned', 'against', 'asked', 'known', 'too', 'have', 'member', 'man', 'everything', 'together', 'knows', 'furthering', 'side', 'may', 'seemed', 'within', 'could',
                      'knew', 'off', 'generally', 'places', 'almost', 'use', 'several', 'lets', 'upon', 'while', 'points', 'second', 'that', 'high', 'find', 'whether', 'members', 'than', 'me', 'quite',
                      'different', 'all', 'always', 'new', 'took', 'downs', 'already', 'everyone', 'mr', 'shall', 'less', 'my', 'becomes', 'were', 'present', 'problems', 'since', 'became', 'behind', 'no',
                      'around', 'and', 'men', 'of', 'today', 'working', 'make', 'says', 'on', 'certainly', 'or', 'interesting', 'any', 'opening', 'until', 'interested', 'backs', 'thought', 'about', 'anywhere',
                      'somewhere', 'worked', 'differently', 'above', 'downing', 'let', 'state', 'thinks', 'fully', 'they', 'grouped', 'began', 'old', 'myself', 'want', 'them', 'then', 'each', 'something', 'pointed',
                      'himself', 'wanted', 'highest', 'must', 'fact', 'another', 'furthered', 'facts', 'two', 'long', 'seem', 'into', 'are', 'does', 'taken', 'came', 'where', 'so', 'gives', 'latest', 'think',
                      'necessary', 'though', 'one', 'many', 'such', 'face', 'oldest', 'showed', 'ask', 'to', 'thing', 'open', 'but', 'through', 'numbers', 'goods', 'needing', 'had', 'either', 'things', 'has', 'up',
                      'newest', 'those', 'us', 'seeming', 'given', 'last', 'might', 'felt', 'this', 'longer', 'once', 'sees', 'everywhere', 'know', 'full', 'higher', 'next', 'away', 'asking', 'needed', 'show',
                      'non', 'we', 'anything', 'not', 'backing', 'interest', 'ends', 'now', 'wants', 'every', 'early', 'thoughts', 'cases', 'again', 'was', 'yet', 'way', 'what', 'backed', 'whole', 'during', 'three',
                      'when', 'put', 'seconds', 'problem', 'her', 'largely', 'far', 'nobody', 'greater', 'between', 'case', 'evenly', 'give', 'still', 'having', 'work', 'come', 'grouping', 'itself', 'toward', 'among',
                      'anyone', 'noone', 'youngest', 'our', 'out', 'across', 'ways', 'get', 'older', 'place', 'more', 'beings', 'mrs', 'puts', 'ended', 'cannot', 'parted', 'sides', 'interests', 'certain', 'first',
                      'small', 'before', 'clearly', 'used', 'him', 'his', 'shows', 'presented', 'only', 'should', 'few', 'from', 'pointing', 'keeps', 'group', 'like', 'kind', 'opened', 'done', 'both', 'important',
                      'most', 'ending', 'parting', 'keep', 'herself', 'seems', 'who', 'here', 'everybody', 'part', 'their', 'why', 'point', 'can', 'general', 'alone', 'along', 'wells', 'said', 'greatest', 'will',
                      'clear', 'saw', 'groups', 'also', 'say', 'enough', 'gets', 'differ', 'really', 'someone', 'uses', 'further', "please"]

    stopWordsDI =  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                      'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                      'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                      'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                      'can', 'will', 'just', 'don', 'should', 'now', "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'm", "you're", "he's", "she's",
                      "it's", "we're", "they're", "i've", "we've", "you've", "they've", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "don't", "doesn't", "didn't", "won't", "wouldn't",
                      "shan't", "shouldn't", "mustn't", "can't", "couldn't", 'cannot', 'could', "here's", "how's", "let's", 'ought', "that's", "there's", "what's", "when's", "where's", "who's", "why's", 'would',
                    'mostly', 'year', 'years', 'areas', 'finds', 'sure', 'thus', 'going', 'younger', 'turning', 'anybody', 'ordering', 'turn', 'turned', 'turning', 'turns', 'see', 'sees','seeming','seemed', 'seems','sees',
                    'seem']

    ngramsStarter = ['and', 'any', 'anyone', 'anything', 'are', 'be', 'best', 'can', 'cannot', 'cant', "can't", 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
                     'done', "don't", 'either', 'else', 'even', 'every', 'for', 'from', 'have', "haven't", "he's", 'is', "isn't", 'it', 'its', "i've", 'just', 'like', 'lots', 'many', 'maybe',
                     'me', 'might', 'more', 'must', 'my', 'never', 'no', 'none', 'not', 'nothing', 'now', 'of', 'on', 'once', 'one', 'only', 'or', 'overly', 'perfectly', 'perhaps', 'probably', 'seemed',
                     'seems', "she's", 'should', 'simply', 'so', 'some', 'somehow', 'something', 'soon', 'start', 'takes', 'tell', 'thank', "that's", 'the', 'their', 'them', 'then', 'there', "there's",
                     'they', "they're", 'this', 'those', 'to', 'too', 'totally', 'tried', 'truly', 'try', 'turns', 'until', 'upon', 'use', 'very', 'wait', 'was', 'well', 'went', 'were', 'whether', 'which',
                     'whole', 'why', 'will', 'wish', "won't", 'would', "wouldn't", 'you', "you'll", 'your', "you're", 'yourself']
