# Translates code from one language to another
#
#   1st arg - code file to translate;
#   2nd - Python key list of original language;
#   3rd - key list of target language;
#   4th - the new file extension
#
# NOTE: The language argument for a lang1 must correspond to a .txt file with
#   the same name as 'lang1Key.txt'.  Otherwise the program will not be able
#   to find the right key mapping.
#

import sys
import string
from unicodedata import category

### MAIN IDEA OF THIS VERSION IS, AFTER REORDERING THE SENTENCE, TOKENIZE EVERYTHING IN AN ARRAY, AND TRANSLATE AFTERWARDS

def swapLineOrder(line):
    # goes through line similarly to main program, but just swaps order
    tokens = list()
    del_quotes = False
    comment = False
    i = 0
    while i < len(line):
        # finding the words
        word = ""
        word_flag = False
        while i<len(line) and is_any_alpha(line[i]):#line[i] not in non_alpha:#(line[i].isalpha()) or (line[i] == '_')):
            word_flag = True
            word += line[i]
            i += 1
        # word must be found, so add it
        if word != '':
            if comment:
                #tokens.append(word)
                tokens.insert(str_start_point, word)
                str_start_point += 1
            elif del_quotes:
                #tokens.append(word)
                tokens.insert(str_start_point, word)
                str_start_point += 1
            else:
                tokens.insert(0,word)
            
        # now write the other separators/operators/etc.
        while (i < len(line) and not is_any_alpha(line[i])):#line[i] in non_alpha):#(not line[i].isalpha()) and line[i] != '_'):
            # for not reordering comments, things in quotes, etc.
            if line[i] == "#" and not comment:
                comment = True
                # no need to reorder anything after this
                # reorder, clean up, and return
                #tokens.reverse()
                #delimiterSwap(tokens)
                tokens.insert(0,line[i])
                str_start_point = 0
                #tokens.insert(0,line[i+1:])
                
                #i = len(line)
                #return "".join(tokens.append(line[i:]))
                
            elif (line[i] == "'" or line[i] == '"') and not comment:
                # check for '''
                if not (i+2 < len(line) and line[i] == "'" and line[i+1] == "'" and line[i+2] == "'"):
                    if del_quotes == True:
                        del_quotes = False
                        tokens.insert(str_start_point, line[i])
                    elif del_quotes == False:
                        del_quotes = True
                        tokens.insert(0, line[i])
                        str_start_point = 1
                else: # otherwise just write the three '''
                    tokens.insert(0,"'''")
                    str_start_point = 1
                    i += 2
            # time to add the item to the token list
            elif (comment) and line[i] != '\n':
                tokens.insert(str_start_point, line[i])
                str_start_point += 1
            elif del_quotes == False:
                # swapping delimiters if necessary
                if line[i] == ')':#'\u202C)\u202C':
                    tokens.insert(0,'(')#'\u202C(\u202C'
                elif line[i] == '(':#'\u202C(\u202C':
                    tokens.insert(0,')')#'\u202C)\u202C'
                elif line[i] == '[':#'\u202C[\u202C':
                    tokens.insert(0,']')#'\u202C]\u202C'
                elif line[i] == ']':#'\u202C]\u202C':
                    tokens.insert(0,'[')#'\u202C[\u202C'
                elif line[i] == '{':#'\u202C{\u202C':
                    tokens.insert(0,'}')#'\u202C}\u202C'
                elif line[i] == '}':#'\u202C}\u202C':
                    tokens.insert(0,'{')#'\u202C{\u202C'
                elif line[i] == '<':#'\u202C<\u202C':
                    tokens.insert(0,'>')#'\u202C>\u202C'
                elif line[i] == '>':#'\u202C>\u202C':
                    tokens.insert(0,'<')#'\u202C<\u202C'
                else:
                    tokens.insert(0,line[i])
                    #print("inserting a '"+str(line[i])+"'")

            else:
                #tokens.append(line[i])
                tokens.insert(str_start_point, line[i])
                str_start_point += 1
            i += 1
    # join the resulting tokens list, return the string
    #print(tokens)
    #result = ''.join(tokens)
    #print('result after swapping:',result)
    #return result
    return tokens
        
###############################

# similar to the swapLineOrder(), but for RTL -> LTR
def swapBack(line):
    # goes through line similarly to main program, but just swaps order
    tokens = list()
    del_quotes = False
    comment = False
    i = len(line) - 1
    while i >= 0:
        # finding the words
        word = ""
        word_flag = False
        while i >= 0 and is_any_alpha(line[i]):#line[i] not in non_alpha:#(line[i].isalpha()) or (line[i] == '_')):
            word_flag = True
            word = line[i] + word
            i -= 1
        # word must be found, so add it
        if word != '':
            if comment:
                #tokens.append(word)
                tokens.insert(str_start_point, word)
                #str_start_point -= 1
            elif del_quotes:
                #tokens.append(word)
                tokens.insert(str_start_point, word)
                #str_start_point -= 1
            else:
                tokens.insert(len(line)-1,word)
            
        # now write the other separators/operators/etc.
        while (i >= 0 and not is_any_alpha(line[i])):#line[i] in non_alpha):#(not line[i].isalpha()) and line[i] != '_'):
            # for not reordering comments, things in quotes, etc.
            if line[i] == "#" and not comment:
                comment = True
                # no need to reorder anything after this
                # reorder, clean up, and return
                #tokens.reverse()
                #delimiterSwap(tokens)
                tokens.insert(len(tokens),line[i])
                str_start_point = len(tokens)
                #tokens.insert(0,line[i+1:])
                
                #i = len(line)
                #return "".join(tokens.append(line[i:]))
                
            elif (line[i] == "'" or line[i] == '"') and not comment:
                if del_quotes == True:
                    del_quotes = False
                    tokens.insert(len(tokens), line[i])
                elif del_quotes == False:
                    del_quotes = True
                    tokens.insert(len(line)-1, line[i])
                    str_start_point = len(tokens)
            # time to add the item to the token list
            elif comment and line[i] != '\n':
                tokens.insert(str_start_point, line[i])
                #str_start_point -= 1
            elif del_quotes == False:
                # swapping delimiters if necessary
                if line[i] == ')':#'\u202C)\u202C':
                    tokens.insert(len(line)-1,'(')#'\u202C(\u202C'
                elif line[i] == '(':#'\u202C(\u202C':
                    tokens.insert(len(line)-1,')')#'\u202C)\u202C'
                elif line[i] == '[':#'\u202C[\u202C':
                    tokens.insert(len(line)-1,']')#'\u202C]\u202C'
                elif line[i] == ']':#'\u202C]\u202C':
                    tokens.insert(len(line)-1,'[')#'\u202C[\u202C'
                elif line[i] == '{':#'\u202C{\u202C':
                    tokens.insert(len(line)-1,'}')#'\u202C}\u202C'
                elif line[i] == '}':#'\u202C}\u202C':
                    tokens.insert(len(line)-1,'{')#'\u202C{\u202C'
                elif line[i] == '<':#'\u202C<\u202C':
                    tokens.insert(len(line)-1,'>')#'\u202C>\u202C'
                elif line[i] == '>':#'\u202C>\u202C':
                    tokens.insert(len(line)-1,'<')#'\u202C<\u202C'
                else:
                    tokens.insert(len(line)-1,line[i])
                    #print("inserting a '"+str(line[i])+"'")
            else:
                #tokens.append(line[i])
                tokens.insert(str_start_point, line[i])
                #str_start_point -= 1
            i -= 1
    # join the resulting tokens list, return the string
    #print(tokens)
    #result = ''.join(tokens)
    #print('result after swapping:',result)
    #return result
    return ''.join(tokens)


Lang1_list = list()
Lang2_list = list()
# determining whether term order needs to be switched (support for right-left languages)
RTL = ['Kurdish','Arabic'] # UPDATE THIS AND uniPython.py AS LANGUAGE LIST GROWS!!!
lang1 = sys.argv[2]
lang2 = sys.argv[3]
orderSwap = False
both_RTL = False
to_RTL = False # this is about direction of swap
if (lang1 not in RTL and lang2 in RTL):
    orderSwap = True
    to_RTL = True
elif (lang1 in RTL and lang2 not in RTL):
    orderSwap = True
    to_RTL = False
elif (lang1 in RTL and lang2 in RTL):
    orderSwap = True
    both_RTL = True
    
# set of non-alphanumeric characters used in Python
# include '_' in this set since key terms may have these instead of ' '
def is_any_alpha(s):
    return (s=='_') or all(category(c)[0] in ["M", "L"] for c in s)
#non_alpha = {' ','\t','\n', '\r', '\b'}
#non_alpha = non_alpha.union(string.punctuation)
#non_alpha.remove('_')

Lang1_file = open('LanguageData/'+sys.argv[2]+'Key.txt','r')
# reading data for first language
line = Lang1_file.readline()
while line != "":
    Lang1_list.append(line.split()[0].strip()) # append without extra '\n'
    line = Lang1_file.readline()
Lang1_file.close()


Lang2_file = open('LanguageData/'+sys.argv[3]+'Key.txt','r')
# reading data for second language
line = Lang2_file.readline()
while line != "":
    Lang2_list.append(line.split()[0].strip()) # append without extra '\n'
    line = Lang2_file.readline()
Lang2_file.close()


# create new py file as the result:
'''
# get rid of .xxpy
new_py_name = ""
for i in str(sys.argv[1]):
    if i == '.':
        break
    else:
        new_py_name += i
        
new_py_name = new_py_name + str(sys.argv[4])
'''
new_py_name = "code2.unipy" # this is simpler for the sake of the demo

new_py = open(new_py_name, 'w')
#new_py = open("test_translation.py","w")

original = open(sys.argv[1],'r')
# now go through the original py file and translate to the new
line = original.readline()
delimeter_quotes = (False,'')
fstring = False
fstring_braces = False
while line != "":
    
    tokenList = list()
    tokenIndex = list()
    
    ### if changing word order, must swap first to avoid complications with key orderings
    if orderSwap == True and (not to_RTL or both_RTL):
        line = swapBack(line)
    comment = False
    i = 0
    while i < len(line): # throughout the entire line
        # finding the words
        #print('line[i] =',line[i])
        word = ""
        # all entries in foreign dict either '_' or alphabetic characters
        #   (foregin font scripts or otherwise)
        word_flag = False
        while i<len(line) and is_any_alpha(line[i]):#line[i] not in non_alpha:#(line[i] == '_' or line[i] not in non_alpha): #((line[i].isalpha()) or (line[i] == '_')):
            #print(line[i])
            # Milind: check for nltk, stopwords, numbers, escape sequences in ASCII instead
            #print("'"+line[i]+"', i =",i)
            word_flag = True
            if (line[i] != '\u202A' and line[i] != '\u202C'):
                word += line[i]
            i += 1

        #print(word)
            
        if word_flag == True:
            #print('word_flag is true')
            # if there is a word, must have encountered a non-alpha or '_',
            #   so it's complete
            # search for the word in the foregin dictionary
            replace_flag = False
            if delimeter_quotes[0] == False and (not comment) or (delimeter_quotes[0] and fstring_braces and not comment):
                for j in range(0, len(Lang1_list)-1):
                    #print(word,'compare to:',Lang1_list[j])
                    if Lang1_list[j] == word:
                        # set replace flag to True and break
                        # write the English version of the word in the new Py file
                        replace_flag = True
                        #print('must be a match in the foreign list')
                        #new_py.write(Lang2_list[j])
                        # Instead of translating now, save the (index,translation) and wait until later
                        tokenList.append(word)
                        #tuple = ((len(tokenList)-1), Lang2_list[j])
                        tokenIndex.append(len(tokenList)-1)
                        break
            
            if replace_flag == False:
                # there was no foreign word to replace,
                #   so just copy to the new Py file
                #new_py.write(word)
                #print('no match found, appending word')
                tokenList.append(word)

        # now write the other separators/operators/etc.
        # if a separator is first in the line, the prior part of loop is skipped
        while (i < len(line) and not is_any_alpha(line[i])):#line[i] in non_alpha):#(not line[i].isalpha()) and line[i] != '_'):
            # for not translating comments, things in quotes, etc.
            if line[i] == "#":
                comment = True
            if not comment and (line[i] == "'" or line[i] == '"'):
                if delimeter_quotes == (True,line[i]):
                    delimeter_quotes = (False,'')
                    fstring = False
                    #print('SO IT DETECTED... on line',line)
                    #print('delimiter quotes now is',False)
                elif delimeter_quotes[0] == False:
                    delimeter_quotes = (True,line[i])
                    if i > 0 and line[i-1]=="f":
                        # this is an f-string
                        fstring = True
                    #print('SO IT DETECTED... on line',line)
                    #print('delimiter quotes now is',delimeter_quotes)
            if not comment and line[i]=='{' and fstring:
                fstring_braces = True
            elif not comment and line[i]=='}' and fstring:
                fstring_braces = False
            #new_py.write(line[i])
            tokenList.append(line[i])
            i += 1
            

    
    for index in tokenIndex:
        word = tokenList[index]
        #print(word)
        for j in range(0, len(Lang1_list)-1):
            if Lang1_list[j] == word:
                #print('found a word:',word)
                replacement = '\u202A'+Lang2_list[j]+'\u202C'
                tokenList[index] = replacement
                #tokenList.pop(index)
                #tokenList.insert(index, replacement)
                break
    
    ### if changing word order to RTL, must swap after to avoid complications with key orderings
    if orderSwap == True and (to_RTL or both_RTL):
        tokenList = swapLineOrder(tokenList)
        
    if len(tokenList) > 0 and tokenList[0]=='\n':
        tokenList.pop(0)
        to_write = ''.join(tokenList) + '\n'

    else:
        to_write = ''.join(tokenList)
    #print('to write:',to_write)
    
    new_py.write(to_write)

    line = original.readline()
    
original.close()
new_py.close()

# ADDED STUFF FOR THE WEB DEMO
# STUFF BEFORE IS JUST THE CodeTranslator.py FILE
new_py = open(new_py_name, 'r')
line = new_py.readline()
to_return = line
while line != '':
    line = new_py.readline()
    to_return += line
print(to_return)

new_py.close()
