res = []
with open("group_tu.txt","r",encoding="utf8") as rg:
    res = [x.split("\n")[0].strip() for x in rg.readlines()]
res.sort()
with open("group_tu2.txt","w",encoding="utf8") as wg:
    for x in res: wg.write(x+"\n")