import React, { Component } from "react"
import Validator from "validator"
import Input from "./../UI/Input/Input"
import Button from "./../UI/Button/Button"
import styles from "./LoginFormContainer.module.css"
import axios from "axios"
class LoginForm extends Component {
    constructor(props) {
        super(props)
        this.state = {
            form: {
                enrollNo: {
                    elementType: "input",
                    inputType: "Number",
                    placeholder: "EnrollNum",
                    value: "",
                    validationRules: {
                        required: true,
                        isNumber: true
                    },
                    valid: false,
                    touched: false
                },
                password: {
                    elementType: "input",
                    inputElement: "password",
                    placeholder: "Password",
                    value: "",
                    validationRules: {
                        required: true,
                    },
                    valid: false,
                    touched: false
                }
            }
        }
    }

    checkValidity(rules, value) {
        let isValid = true
        Object.keys(rules).forEach(rule => {
            if (rule == "required") {
                isValid = isValid && !Validator.isEmpty(Validator.ltrim(Validator.rtrim(value)))
            } else if (rule == "minlen") {
                isValid = isValid && Validator.isLength(value, rules[rule])
            } else if (rule == "maxlen") {
                isValid = isValid && Validator.isLength(value, 0, rules[rule] + 1)
            } else if (rule == "isEmail") {
                isValid = isValid && Validator.isEmail(value)
            } else if (rule == "isNumber") {
                isValid = isValid && Validator.isNumeric(value)
            }
        })
        return isValid
    }

    inputChangedHandler(event, inputIdentifier) {
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
        for (let inputIdentifier in updatedForm) {
            formIsValid = updatedForm[inputIdentifier].valid && formIsValid
        }

        //set the from in the state to the updated form and also set the validity status of yhe form
        this.setState({
            form: updatedForm,
            formIsValid: formIsValid
        })
    }

    loginHandler = async(event) => {
        try {
            event.preventDefault()
            let formData = {}
            for (let formElementIdentifier in this.state.form) {
                formData[formElementIdentifier] = this.state.form[formElementIdentifier].value;
            }
            console.log(formData)
            const res = await axios({
                method: "POST",
                url: "http://127.0.0.1:9000/emotion-detector/user/login",
                data: formData
            })
            if (res.data.status == "success") {
                console.log(res.data)
                this.props.setUpAuthHandler(res.data.user[0])
                alert("LoggedIn successfully")
            }
        } catch (error) {
            console.log(error)
        }
    }
    render() {
        const form = Object.keys(this.state.form).map(inputField => ( <
            Input elementType = { this.state.form[inputField].elementType }
            inputType = { this.state.form[inputField].inputType }
            placeholder = { this.state.form[inputField].placeholder }
            value = { this.state.form[inputField].value }
            valid = { this.state.form[inputField].valid }
            changed = {
                (event) => this.inputChangedHandler(event, inputField) }
            touched = { this.state.form[inputField].touched }
            />
        ))
        return ( <
            form onSubmit = { this.loginHandler }
            className = { styles.LoginFormContainer } > { form } <
            Button classes = {
                ["Submit"] }
            disabled = {!this.state.formIsValid } > Submit < /Button> <
            /form>
        )
    }
}

export default LoginForm