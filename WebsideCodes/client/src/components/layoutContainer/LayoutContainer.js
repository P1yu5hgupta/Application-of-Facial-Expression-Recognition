import React, { Component } from "react"
import styles from "./LayoutContainer.module.css"
import Toolbar from "./../Toolbar/Toolbar"
import Auxillary from "./../../hocs/Auxillary/Auxillary"
import SignUpForm from "./../formsContainers/SignupFormContainer"
import LoginForm from "./../formsContainers/LoginFormContainer"
import ProjectDesc from "./../UI/ProjectDesc/ProjectDesc"
import Name from "./../Profile/name"
import Cards from "./../Cards/Cards"

class LayoutContainer extends Component {
    constructor(props) {
        super(props);
        this.state = {
            auth: {
                status: false,
                data: null
            },
            formType: "login"
        }
    }
    changeFormTypeHandler = (value) => {
        this.setState({
            formType: value
        })
    }
    setUpAuthHandler = (data) => {
        this.setState({
            auth: {
                status: true,
                data: data
            }
        })
    }
    render() {
        return ( <
            div className = { styles.LayoutContainer } >
            <
            Toolbar changeFormTypeHandler = { this.changeFormTypeHandler }
            authData = { this.state.auth }
            /> <
            ProjectDesc / > {
                this.state.auth.status ? < Cards data = { this.state.auth.data }
                />:
                this.state.formType == "signup" ?
                <
                SignUpForm setUpAuthHandler = { this.setUpAuthHandler }
                />: <
                LoginForm setUpAuthHandler = { this.setUpAuthHandler }
                />
            } <
            /div>
        )
    }
}

export default LayoutContainer;