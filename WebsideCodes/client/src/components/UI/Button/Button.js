import React from "react"
import styles from "./Button.module.css"

const Button = (props)=>{
  let classes = [styles.Button]
  props.classes.forEach(element => {
    classes.push(styles[element])
  });
  return(
    <div>
      <button className = {classes.join(" ")} onClick = {props.clicked}>{props.children}</button>
    </div>
  )
}
export default Button