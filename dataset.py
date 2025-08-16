dataset=[]
with open("cat.txt",'r',encoding='utf=8') as file:
    dataset=file.readlines()
  
    print(f'loaded {len(dataset)} enteries')