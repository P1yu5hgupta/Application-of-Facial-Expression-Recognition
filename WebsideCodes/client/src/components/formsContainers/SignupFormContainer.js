import React, {Component} from "react";
import Button from "./../UI/Button/Button";
import styles from "./SignupFormContainer.module.css";
import axios from "axios";
import Input from "./../UI/Input/Input";
import Validator from "validator"

class SignupFormContainer extends Component{
  constructor(props){
    super(props)
    this.state = {
      form:{
        name: {
          elementType: "input",
          inputType: "text",
          placeholder: "Your Name",
          value: "",
          validationRules:{
            required: true
          },
          valid: false,//whether value is false or correct
          touched: false
        },
        enrollNo:{
          elementType: "input",
          inputType: "number",
          placeholder: "Your Enroll No",
          valid: "false",
          validationRules:{
            required: true,
            isNumber: true
          },
          valid: false,
          touched: false
        },
        email:{
          elementType: "input",
          inputType: "email",
          placeholder: "Your Email",
          value:" ",
          valid: false,
          touched: false,
          validationRules:{
            required: true,
            isEmail: true
          }
        },
        password:{
          elementType: "input",
          inputType: "password",
          placeholder: "Password",
          value: "",
          valid: false,
          touched: false,
          validationRules:{
            required: true,
          }
        }
      },
      formIsValid: false,//this property can be used to check whether we can submit the form or not
      loading: false
    }
  }

  checkValidity(rules, value){
    let isValid = true
    Object.keys(rules).forEach(rule => {
      if(rule == "required"){
        isValid = isValid && !Validator.isEmpty(Validator.ltrim(Validator.rtrim(value)))
      }else if(rule == "minlen"){
        isValid = isValid && Validator.isLength(value, rules[rule])
      }else if(rule == "maxlen"){
        isValid = isValid && Validator.isLength(value, 0, rules[rule]+1)
      }else if(rule == "isEmail"){
        isValid = isValid && Validator.isEmail(value)
      }else if(rule == "isNumber"){
        isValid = isValid && Validator.isNumeric(value)
      }
    })
    return isValid
  }
  inputChangedHandler(event, inputIdentifier){
    //make an instance fo the form
    const updatedForm = {
      ...this.state.form
    }
    //console.log(updatedForm)
    //console.log(inputIdentifier)
    //get the element of yhe form which needs to be updated 
    const updatedFormElement = updatedForm[inputIdentifier]

    //set the new value
    updatedFormElement.value = event.target.value
    //check the validity of the new value
    updatedFormElement.valid = this.checkValidity(updatedFormElement.validationRules, updatedFormElement.value)

    updatedFormElement.touched = true
    
    //set the element of the form to the updated form
    updatedForm[inputIdentifier] = updatedFormElement
    
    //check the validity of the form
    let formIsValid = true
    for (let inputIdentifier in updatedForm){
      formIsValid = updatedForm[inputIdentifier].valid && formIsValid
    }

    //set the from in the state to the updated form and also set the validity status of yhe form
    this.setState({
      form:updatedForm,
      formIsValid: formIsValid
    })

  }

  signupHandler = async (event) => {
    try{
      event.preventDefault()
      let formData = {}
      for (let formElementIdentifier in this.state.form) {
        formData[formElementIdentifier] = this.state.form[formElementIdentifier].value;
    }
    console.log(formData)
      const res = await axios({
        method: "POST",
        url: "http://127.0.0.1:9000/emotion-detector/user/signup",
        data: formData
      })
      if(res.data.status == "success"){
        console.log(res.data.newUser)
        this.props.setUpAuthHandler(res.data.newUser)
        alert("signed up successfully")        
      }
    }catch(error){
      console.log(error)
    }
  }

  render(){
    const form = Object.keys(this.state.form).map(inputField => (
      <Input elementType = {this.state.form[inputField].elementType}
      inputType = {this.state.form[inputField].inputType}
      placeholder = {this.state.form[inputField].placeholder}
      value = {this.state.form[inputField].value}
      valid = {this.state.form[inputField].valid}
      changed = {(event) => this.inputChangedHandler(event, inputField)}
      touched = {this.state.form[inputField].touched}
      />
    ))
    return(
      <form onSubmit = {this.signupHandler} className = {styles.SignupFormContainer}>
        {form}
        <Button classes = {["Submit"]} disabled = {!this.state.formIsValid}>Submit</Button>
      </form>
    )
  }
}

export default SignupFormContainer;