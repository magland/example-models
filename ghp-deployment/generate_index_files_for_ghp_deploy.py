#!/usr/bin/env python3
"""
Script to automatically generate index.md files for all directories.

This script recursively searches through the project and creates index.md files
for all directories. For directories containing .stan and .data.json file pairs,
it creates files with stan-playground embed code for viewing the models in Stan
Playground. For other directories, it creates simple index files with links to
all subdirectories and files. Always overwrites existing index.md files.
"""

from pathlib import Path


def find_stan_data_pairs(directory):
    """
    Find all .stan files and their corresponding .data.json files (if they exist) in a directory.
    
    Args:
        directory (Path): Directory to search in
        
    Returns:
        list: List of tuples (stan_file, data_file) where data_file can be None if no corresponding .data.json exists
    """
    pairs = []
    stan_files = list(directory.glob("*.stan"))
    
    for stan_file in stan_files:
        # Get the base name without extension
        base_name = stan_file.stem
        # Look for corresponding .data.json file
        data_file = directory / f"{base_name}.data.json"
        
        if data_file.exists():
            pairs.append((stan_file.name, data_file.name))
        else:
            # Include .stan file even without corresponding .data.json
            pairs.append((stan_file.name, None))
            
    return pairs


def has_stan_data_pairs_recursive(directory):
    """
    Recursively check if a directory or any of its subdirectories contains .stan/.data.json pairs.
    
    Args:
        directory (Path): Directory to search in
        
    Returns:
        bool: True if any .stan/.data.json pairs are found in the directory tree, False otherwise
    """
    # Check if the current directory has pairs
    if find_stan_data_pairs(directory):
        return True
    
    # Recursively check all subdirectories
    for item in directory.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Skip common build/cache directories
            if item.name not in ['node_modules', '__pycache__', '.git', '_site']:
                if has_stan_data_pairs_recursive(item):
                    return True
    
    return False


def generate_directory_content(directory):
    """
    Generate the content for an index.md file for directories without Stan pairs.
    
    Args:
        directory (Path): Directory containing the files
        
    Returns:
        str: Generated index.md content
    """
    content = f"# {directory.name}\n\n"
    
    # Get all subdirectories and categorize them
    subdirs_with_stan = []
    subdirs_other = []
    
    for item in directory.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if item.name not in ['node_modules', '__pycache__', '.git', '_site']:
                # Check if this subdirectory contains .stan/.data.json pairs recursively
                subdir_path = directory / item.name
                if has_stan_data_pairs_recursive(subdir_path):
                    subdirs_with_stan.append(item.name)
                else:
                    subdirs_other.append(item.name)
    
    # Add subdirectories with Stan files section
    if subdirs_with_stan:
        subdirs_with_stan.sort()
        content += "## Subdirectories with Stan Files\n\n"
        for subdir in subdirs_with_stan:
            content += f"- [{subdir}](./{subdir}/)\n"
        content += "\n"
    
    # Add other subdirectories section
    if subdirs_other:
        subdirs_other.sort()
        content += "## Other Subdirectories\n\n"
        for subdir in subdirs_other:
            content += f"- [{subdir}](./{subdir}/)\n"
        content += "\n"
    
    # Get all files in the directory, excluding index.md itself
    all_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name != 'index.md':
            all_files.append(file_path.name)
    
    if all_files:
        all_files.sort()
        content += "## Files\n\n"
        for filename in all_files:
            content += f"- [{filename}](./{filename})\n"
    
    return content


def generate_index_content(pairs, directory):
    """
    Generate the content for an index.md file based on stan/data pairs.
    
    Args:
        pairs (list): List of tuples (stan_file, data_file) where data_file can be None
        directory (Path): Directory containing the files
        
    Returns:
        str: Generated index.md content
    """
    content = '<script src="https://stan-playground.flatironinstitute.org/stan-playground-embed.js"></script>\n\n'
    
    for stan_file, data_file in pairs:
        # Add heading with the filename including .stan extension
        base_name = Path(stan_file).stem
        content += f'## {base_name}.stan\n\n'
        
        content += f'<stan-playground-embed\n'
        # Only add data attribute if data_file exists
        if data_file is not None:
            content += f'    data="./{data_file}"\n'
        content += f'    stan="./{stan_file}"\n'
        content += f'>\n'
        content += f'<iframe width="100%" height="800px"></iframe>\n'
        content += f'</stan-playground-embed>\n'
        
        # Add spacing between multiple embeds
        if len(pairs) > 1 and (stan_file, data_file) != pairs[-1]:
            content += '\n'
    
    # Add links to all files in the directory
    content += '\n## Files\n\n'
    
    # Get all files in the directory, excluding index.md itself
    all_files = []
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name != 'index.md':
            all_files.append(file_path.name)
    
    # Sort files for consistent ordering
    all_files.sort()
    
    for filename in all_files:
        content += f'- [{filename}](./{filename})\n'
    
    return content


def process_directory(directory, stats):
    """
    Process a single directory and create index.md file.
    
    Args:
        directory (Path): Directory to process
        stats (dict): Statistics dictionary to update
    """
    pairs = find_stan_data_pairs(directory)
    index_path = directory / "index.md"
    
    # Determine if this index.md existed before
    existed_before = index_path.exists()
    
    if pairs:
        print(f"Found {len(pairs)} Stan/data pair(s) in {directory}")
        for stan_file, data_file in pairs:
            print(f"  - {stan_file} + {data_file}")
        
        # Generate content with Stan playground embeds
        content = generate_index_content(pairs, directory)
    else:
        # Generate content for directory without Stan pairs
        content = generate_directory_content(directory)
    
    # Always write the index.md file (overwrite if exists)
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if existed_before:
            print(f"  ✓ Updated {index_path}")
            stats['updated'] += 1
        else:
            print(f"  ✓ Created {index_path}")
            stats['created'] += 1
            
    except Exception as e:
        print(f"  ✗ Error writing {index_path}: {e}")
        stats['errors'] += 1


def main():
    """Main function to run the script."""
    print("Stan Index Generator")
    print("===================")
    print("Creating index.md files for all directories...")
    print()
    
    # Start from the current working directory
    root_dir = Path(".")
    
    # Statistics tracking
    stats = {
        'created': 0,
        'updated': 0,
        'errors': 0,
        'directories_scanned': 0
    }
    
    # Process the root directory first
    stats['directories_scanned'] += 1
    process_directory(root_dir, stats)
    
    # Recursively walk through all directories
    for current_dir in root_dir.rglob("*"):
        if current_dir.is_dir():
            # Skip hidden directories and common build/cache directories
            if any(part.startswith('.') for part in current_dir.parts):
                continue
            if any(part in ['node_modules', '__pycache__', '.git', '_site'] for part in current_dir.parts):
                continue
                
            stats['directories_scanned'] += 1
            process_directory(current_dir, stats)
    
    # Print summary
    print()
    print("Summary")
    print("=======")
    print(f"Directories scanned: {stats['directories_scanned']}")
    print(f"Index files created: {stats['created']}")
    print(f"Index files updated: {stats['updated']}")
    print(f"Errors encountered: {stats['errors']}")
    
    if stats['created'] + stats['updated'] > 0:
        print(f"\n✓ Successfully processed {stats['created'] + stats['updated']} directories")
    else:
        print("\nNo directories found to process.")


if __name__ == "__main__":
    main()
