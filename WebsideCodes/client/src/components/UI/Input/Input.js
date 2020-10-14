import React from "react";
import styles from "./Input.module.css";
import Auxillary from "./../../../hocs/Auxillary/Auxillary"

const input = (props) => {
    let inputElement = null //we will use this var to store the final return value
    const inputClasses = [styles.InputElement]

    console.log(props.valid, props.touched)
    if (!props.valid && props.touched) {
        inputClasses.push(styles.Invalid)
            //for adding an invalid classes that can help user to know what is wrong
    }

    switch (props.elementType) {
        case ("input"):
            inputElement = < input
            className = { inputClasses.join("  ") }
            type = { props.inputType }
            placeholder = { props.placeholder }
            onChange = { props.changed }
            />
            break
        case ("textarea"):
            inputElement = < textarea
            className = { inputClasses.join("  ") }
            type = { props.inputType }
            placeholder = { props.placeholder }
            value = { props.value }
            onChange = { props.changed }
            />
            break
        case ("select"):
            inputElement = < select
            className = { styles.join(" ") }
            value = { props.value }
            onChange = { props.changed } > {
                    props.elementConfig.options.map(option => ( <
                        option key = { option.value }
                        value = { option.value } > { option.displayValue } <
                        /option>
                    ))
                } <
                /select>
            break
        default:
            inputElement = < input
            className = { inputClasses.join("  ") }
            type = { props.inputType }
            placeholder = { props.placeholder }
            value = { props.value }
            onChange = { props.changed }
            />
    }

    return ( <
        Auxillary > { inputElement } <
        /Auxillary>
    )

}

export default input