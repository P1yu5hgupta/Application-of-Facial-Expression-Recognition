const mongoose = require("mongoose")
const validator = require("validator")
const userScheema = mongoose.Schema({
  name:{
    type: String,
    required: true,
  },
  email:{
    type: String,
    required: true,
    validate: [validator.isEmail, "Please provide a valid email address"]
  },
  enrollNo:{
    type: String,
    required: true,
    validate: [validator.isNumeric, "Please provide a valid enroll num"]
  },
  password:{
    type: String,
    required: true,
    minlength: 8
  },
  emotion:{
    type: Array
  }
})

const User = mongoose.model("User", userScheema)
module.exports = User