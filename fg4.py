# Complete the function below.

def getkey(item):
    return item[0]
def arrange(sentence):
    dict1 = []
    num = []
    str1 = ''
    len1 = len(sentence) - 1
    sentence = sentence.lower()
    words = sentence[0: len1].split(' ')
    for i in words:
        dict1.append((len(i),i))

    for j in sorted(dict1, key = getkey):

        str1 += j[1] + " "
    str2 = str1[0].upper()
    str3 = str2 + str1[1:len(str1) - 1] + '.'
    return str3


k = arrange('The the the the.')
print(k)