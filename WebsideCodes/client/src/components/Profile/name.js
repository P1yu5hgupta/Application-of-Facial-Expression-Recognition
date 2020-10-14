import React from "react"
import styles from "./name.module.css"

const name = (props)=>{
  return(
    <div className = {styles.Name}>
      {props.children}
    </div>
  )
}

export default name