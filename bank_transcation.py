class Bank:
    def __init__(self,bal,acc):
        self.balance = bal
        self.account_number = acc
    def debit(self,amount):
        self.balance-=amount
        print("amount debited: ",amount)
    def credit(self,amount):
        self.balance+=amount
        print("amount credited: ",amount)
        print("New balance is: ",self.balance)
    def total(self):
        return self.balance
ac1=Bank(10000, 123456789)
print(ac1.balance)
print(ac1.account_number)
ac1.debit(1000)
ac1.credit(20000)
