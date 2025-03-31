import React, {useState, useEffect} from 'react';
import { useRouter } from 'next/router';
import { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async (context) => {
  const { email } = context.query;
  if (!email) return { props: { user: null } };

  const res = await fetch(`http://localhost:3000/api/loadArticles`); // TODO: change on deployment
  let articleProp = await res.json();
  const articles = articleProp['message'];

  return { props: { articles }};
};



export default function Dashboard({articles}) {
    // Define state with TypeScript types
    const router = useRouter();
    const { email } = router.query; 
    const articleArray = JSON.parse(articles);
    const [annotationStatus, setAnnotationStatus] = useState({});
    const user = email.split("@")[0];


    useEffect(() => {
        const checkAnnotations = async () => {
            const statuses = {};
            for (const article of articleArray) {
                statuses[article] = await isAnnotated(article);
            }
            setAnnotationStatus(statuses);
        };
        
        if (email) {
            checkAnnotations();
        }
    }, []); // just does it once?
    
    
    const isAnnotated = async (article) => {

        try {
            const res = await fetch(`/api/loadSelections/?article=${article}&user=${user}`);
            if (res.ok) {
                return true;
            } 
            return false;
        } catch (error) {
            return false;
        }
    }

    const openArticle = (article) => {
        router.push(`/article_annotation/?email=${email}&article=${article}`); 
    };

    // if is annotated, then preface with a checkmark

    return (
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <h1 style={{ fontSize: "1.25rem", fontWeight: "bold", textAlign: "center" }}>{user}'s Annotations</h1>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
    
                {articleArray.map((article, index) => (
                    <div style={{ backgroundColor: "#f9f9f9", margin: "1rem", padding: "1rem", borderRadius: "8px", width: "90%", border: "1px solid #ddd",}}>
                        
                        <button onClick={() => openArticle(article)}>
                            {annotationStatus[article] && <span>&#10004;</span>} {article}
                        </button>
                    </div>
                ))}
    
            </div>
        </div>
    );
    
};