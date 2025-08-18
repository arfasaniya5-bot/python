class complex:
    def __init__(self, real, img):
        self.real=real
        self.img=img
    def shownumber(self):
        print(self.real,"i","+",self.img,"j")
    def __add__(self,obj):
        newreal=self.real+obj.real
        newimg=self.img+obj.img
        return complex(newreal,newimg)
num1 = complex(1,7)
num1.shownumber()
num2 = complex(2,3)
num2.shownumber()
num3=num1+num2
num3.shownumber()
