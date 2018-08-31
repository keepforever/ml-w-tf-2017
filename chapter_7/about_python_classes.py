import datetime
# the simplest class
class SimpleUser: 
    #pass is a way to type a line that does nothing
    pass

# envoking class to create a 'SimpleUser' instance/object
user1 = SimpleUser()

# to attach data to class instance
user1.first_name = "Brian"
user1.last_name = "Cilenti"
# first_name and last_name are reffered to as fields. 

class User: 
    # init method, aka constructor
    def __init__(self, full_name, birthday):
        # doc string, gives info about class
        """ 
        some info about this class. 
        """
        # self is a reference to new obj being created.
        #store these values to fields in the object 
        
        # store the passed in full_name arg as a value called 'name'
        self.name = full_name
        self.birthday = birthday #yyyymmdd format

        # perform changes on input data to create more fields

        # break full name into array ['first', 'last]
        name_pieces = full_name.split(" ") # cut whenever encounter a " "
        self.first_name = name_pieces[0]
        self.last_name = name_pieces[1]
    def age(self):
        """ Return the age of the user in years"""
        # get todays date
        today = datetime.date(2001, 5, 12)
        # cutting up the input birthday string via slicing
        yyyy = int(self.birthday[0:4]) # first four chars of string
        mm = int(self.birthday[4:6])
        dd = int(self.birthday[6:8])
        dob = datetime.date(yyyy, mm, dd)
        # yeilds time delta object with a  field called days
        age_in_days = (today - dob).days 
        age_in_years = age_in_days / 365
        return(age_in_years)


user2 = User(full_name='Brian Cilenti', birthday="19841028")

print(user2.name, user2.birthday, user2.first_name, user2.last_name)

# print contents of doc string and some other meta. 
help(User)

# try out User class' age method


print(user2.age())
