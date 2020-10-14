const mongoose = require("mongoose")
//mongoose is required for connection to the database
const app = require("./App")
//now we need to connect app to thee database
const dotenv = require("dotenv")
//Dotenv is a zero-dependency module that loads environment variables from a .env file into process.env
dotenv.config({ path: './config.env' });

const DB = process.env.DATABASE.replace(
  "<password>",
  process.env.DATABASE_PASSWORD
)

//getting database string

mongoose.connect(DB, {
  useNewUrlParser: true,
  useCreateIndex: true,
  useFindAndModify: false
}).then(()=>console.log("DB connection successfull!"))

//connecting to the database

const port = 9000

//creating a server
app.listen(port, ()=>{
  console.log("App is running on port ", port)
})