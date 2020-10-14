import React from "react";
import styles from "./Toolbar.module.css";
import Button from "../UI/Button/Button";
import Aux from "../../hocs/Auxillary/Auxillary";
import Name from "./../Profile/name"

const Toolbar = (props) => {
    return ( <
        div className = { styles.Toolbar } > {
            props.authData.status ?
            <
            Name > { props.authData.data.enrollNo } < /Name>:
            ( < Aux > < Button clicked = {
                    () => props.changeFormTypeHandler("login")
                }
                classes = {
                    ["Auth"]
                } > Login < /Button> <
                Button clicked = {
                    () => props.changeFormTypeHandler("signup")
                }
                classes = {
                    ["Auth"]
                } >
                Signup < /Button></Aux > )
        } <
        /div>
    )
}
export default Toolbar;